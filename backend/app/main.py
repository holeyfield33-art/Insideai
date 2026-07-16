"""InsideAI backend — FastAPI app that streams real transformer internals
over a WebSocket while a HuggingFace model generates text token by token.

Protocol (client -> server):
    {"type": "generate", "prompt": str, "max_new_tokens"?, "temperature"?,
     "top_k"?, "top_p"?, "seed"?, "speed"?}
    {"type": "stop"}
    {"type": "ping"}

Server -> client event types:
    model_info, generation_start, tokenize, embeddings, positional,
    layer_start, attention, ffn, layer_end, logits, sampled, step_end,
    generation_end, error, pong
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .model.engine import GenerationParams, engine
from .streaming.protocol import step_events

logger = logging.getLogger("insideai")

# One forward pass at a time across all connections — a laptop CPU thrashes
# if two traces run concurrently, and interleaving per-step keeps things fair.
ENGINE_LOCK = asyncio.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Loading %s (first run downloads the checkpoint)...", settings.model_name)
    await asyncio.to_thread(engine.load)
    logger.info("Model ready in %.1fs", engine.load_seconds)
    yield


app = FastAPI(title="InsideAI backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "model_loaded": engine.loaded}


@app.get("/model")
async def model_info() -> dict:
    return engine.model_info()


class Session:
    """One websocket connection: serialized sends + at most one generation task."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.task: asyncio.Task | None = None
        self._send_lock = asyncio.Lock()

    async def send(self, payload: dict) -> None:
        async with self._send_lock:
            await self.ws.send_text(json.dumps(payload, separators=(",", ":")))

    def cancel(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()

    async def join_cancelled(self) -> None:
        if self.task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.task
        self.task = None


async def run_generation(session: Session, msg: dict) -> None:
    prompt = str(msg.get("prompt", ""))
    if not prompt.strip():
        await session.send({"type": "error", "message": "Prompt is empty."})
        return

    params = GenerationParams.from_payload(msg)
    chat_mode = str(msg.get("chat_mode", "auto"))
    try:
        speed = float(msg.get("speed", 1.0))
    except (TypeError, ValueError):
        speed = 1.0
    speed = max(0.0, min(3.0, speed))

    started = time.time()
    text = ""
    steps_done = 0
    reason = "max_tokens"
    cache = None
    # Everything from encoding onwards lives inside the try: a failure at any
    # point must surface as an error event, never as a silently dead task.
    try:
        ids, truncated, templated = await asyncio.to_thread(
            engine.encode_prompt, prompt, chat_mode
        )
        prompt_len = int(ids.shape[1])
        generator = engine.make_generator(params.seed)

        await session.send(
            {
                "type": "generation_start",
                "prompt": prompt,
                "prompt_tokens": prompt_len,
                "truncated": truncated,
                "template": templated,
                "speed": speed,
                "params": {
                    "max_new_tokens": params.max_new_tokens,
                    "temperature": params.temperature,
                    "top_k": params.top_k,
                    "top_p": params.top_p,
                    "repetition_penalty": params.repetition_penalty,
                    "seed": params.seed,
                    "strategy": params.describe(),
                },
            }
        )

        for step in range(params.max_new_tokens):
            async with ENGINE_LOCK:
                if step == 0:
                    trace, cache = await asyncio.to_thread(
                        engine.trace_prefill, ids, params, generator
                    )
                else:
                    trace, cache = await asyncio.to_thread(
                        engine.trace_decode, ids, cache, params, generator, prompt_len
                    )

            for event, delay in step_events(trace, step, engine.n_layer):
                await session.send(event)
                if delay > 0 and speed > 0:
                    await asyncio.sleep(delay * speed)

            chosen = trace["sampled"]
            ids = torch.cat(
                [ids, torch.tensor([[chosen["id"]]], device=ids.device)], dim=1
            )
            steps_done = step + 1
            text = engine.tokenizer.decode(ids[0][prompt_len:].tolist())
            await session.send(
                {
                    "type": "step_end",
                    "step": step,
                    "token": chosen,
                    "text": text,
                    "total_seq": int(ids.shape[1]),
                    "compute_ms": trace["compute_ms"],
                }
            )

            if chosen["is_eos"]:
                reason = "eos"
                break
            if ids.shape[1] >= engine.n_positions - 1:
                reason = "context_limit"
                break

        duration = time.time() - started
        await session.send(
            {
                "type": "generation_end",
                "text": text,
                "steps": steps_done,
                "reason": reason,
                "duration_s": round(duration, 2),
                "tokens_per_s": round(steps_done / duration, 2) if duration > 0 else None,
            }
        )
    except asyncio.CancelledError:
        with contextlib.suppress(Exception):
            await session.send(
                {
                    "type": "generation_end",
                    "text": text,
                    "steps": steps_done,
                    "reason": "cancelled",
                    "duration_s": round(time.time() - started, 2),
                    "tokens_per_s": None,
                }
            )
        raise
    except Exception as exc:  # surfaced to the client, logged with traceback
        logger.exception("generation failed")
        with contextlib.suppress(Exception):
            await session.send(
                {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
            )


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    session = Session(ws)
    await session.send({"type": "model_info", **engine.model_info()})
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await session.send({"type": "error", "message": "Invalid JSON."})
                continue

            msg_type = msg.get("type")
            if msg_type == "generate":
                session.cancel()
                await session.join_cancelled()
                session.task = asyncio.create_task(run_generation(session, msg))
            elif msg_type == "stop":
                session.cancel()
            elif msg_type == "ping":
                await session.send({"type": "pong", "t": time.time()})
            else:
                await session.send(
                    {"type": "error", "message": f"Unknown message type: {msg_type!r}"}
                )
    except WebSocketDisconnect:
        pass
    finally:
        session.cancel()
