"""InsideAI backend — FastAPI app that streams real transformer internals
over a WebSocket while a HuggingFace model generates text token by token.

Protocol (client -> server):
    {"type": "generate", "prompt": str, "max_new_tokens"?, "temperature"?,
     "top_k"?, "top_p"?, "seed"?, "speed"?}
    {"type": "replay", "id": str, "speed"?}
    {"type": "stop"}
    {"type": "ping"}

Server -> client event types:
    server_mode, model_info, generation_start, tokenize, embeddings, positional,
    layer_start, attention, ffn, layer_end, logits, sampled, step_end,
    generation_end, error, pong

Replay re-emits a recorded run's events over this same protocol, byte-for-byte,
so the frontend renders a recording exactly as it would a live run (no
special-case paths). See app/recording.py. Replay loads no model.
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
from .recording import RunRecorder, list_runs, load_run_events
from .streaming.protocol import base_delay, step_events

logger = logging.getLogger("insideai")

# One forward pass at a time across all connections — a laptop CPU thrashes
# if two traces run concurrently, and interleaving per-step keeps things fair.
ENGINE_LOCK = asyncio.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if settings.skip_model_load:
        # Replay-only mode: no checkpoint is loaded. Recorded runs still list
        # and replay over the WebSocket; live generation is refused.
        logger.info("INSIDEAI_SKIP_MODEL set — starting in replay-only mode (no model).")
        yield
        return
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
    return {"status": "ok", "model_loaded": engine.loaded, "replay_only": not engine.loaded}


@app.get("/model")
async def model_info() -> dict:
    return engine.model_info()


@app.get("/runs")
async def runs() -> list[dict]:
    """Recorded runs available for replay: [{id, prompt, model, timestamp, steps}]."""
    return list_runs()


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
    if not engine.loaded:
        await session.send(
            {
                "type": "error",
                "message": "No model loaded — this backend is in replay-only mode. "
                "Pick a recorded run to replay.",
            }
        )
        return

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

    # Opt-in recording (INSIDEAI_RECORD): a passive tap on the exact events
    # sent below. `emit` sends then records, so the recording is the live
    # stream verbatim. Recorder is None when recording is off — zero overhead.
    recorder: RunRecorder | None = None
    effective_seed = engine.resolve_seed(params.seed)
    if settings.record:
        recorder = RunRecorder.start(
            model_info=engine.model_info(),
            prompt=prompt,
            params=params,
            effective_seed=effective_seed,
            chat_mode=chat_mode,
            speed=speed,
        )

    async def emit(event: dict) -> None:
        await session.send(event)
        if recorder is not None:
            recorder.record(event)

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
        generator = engine.make_generator(effective_seed)

        await emit(
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
                await emit(event)
                if delay > 0 and speed > 0:
                    await asyncio.sleep(delay * speed)

            chosen = trace["sampled"]
            ids = torch.cat(
                [ids, torch.tensor([[chosen["id"]]], device=ids.device)], dim=1
            )
            steps_done = step + 1
            text = engine.tokenizer.decode(ids[0][prompt_len:].tolist())
            await emit(
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
        await emit(
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
        reason = "cancelled"
        with contextlib.suppress(Exception):
            await emit(
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
        reason = "error"
        logger.exception("generation failed")
        with contextlib.suppress(Exception):
            await session.send(
                {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
            )
    finally:
        if recorder is not None:
            recorder.finalize(text=text, steps=steps_done, reason=reason)


async def replay_run(session: Session, msg: dict) -> None:
    """Re-emit a recorded run's events over the same WebSocket protocol.

    No model is touched: the events are read from disk and streamed verbatim,
    honoring the client's speed control via the shared `base_delay` pacing. The
    frontend can't tell a replayed event from a live one — that's the point —
    so the client tags the stream as a replay on its side and shows the REPLAY
    banner it requested.
    """
    run_id = str(msg.get("id", ""))
    try:
        speed = float(msg.get("speed", 1.0))
    except (TypeError, ValueError):
        speed = 1.0
    speed = max(0.0, min(3.0, speed))

    try:
        events = await asyncio.to_thread(load_run_events, run_id)
    except FileNotFoundError:
        await session.send({"type": "error", "message": f"No such recorded run: {run_id!r}"})
        return
    except Exception as exc:
        logger.exception("failed to load run %s", run_id)
        await session.send({"type": "error", "message": f"Could not read run: {exc}"})
        return

    n_layer = 0
    for event in events:
        if event.get("type") == "model_info":
            n_layer = int(event.get("n_layer", n_layer) or 0)
        await session.send(event)
        delay = base_delay(event, n_layer)
        if delay > 0 and speed > 0:
            await asyncio.sleep(delay * speed)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    session = Session(ws)
    # When a model is loaded, model_info stays the first frame (unchanged live
    # protocol). In replay-only mode there's no model to describe, so only the
    # server_mode frame is sent — the recorded model_info arrives as the first
    # replayed event instead. server_mode always follows so the client learns
    # whether live generation is available.
    if engine.loaded:
        await session.send({"type": "model_info", **engine.model_info()})
    await session.send({"type": "server_mode", "live": engine.loaded})
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
            elif msg_type == "replay":
                session.cancel()
                await session.join_cancelled()
                session.task = asyncio.create_task(replay_run(session, msg))
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
