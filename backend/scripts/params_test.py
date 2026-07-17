"""Parametric behavior test: proves the generation settings actually work.

Runs against a live backend and asserts, from observed behavior:
  * max_new_tokens is respected
  * temperature 0 -> strictly greedy (every sampled token has rank 1)
  * top_k = 1 -> rank 1 even at high temperature
  * a fixed seed reproduces the exact same token ids twice
  * chat mode applies the template, raw mode doesn't
  * repetition_penalty / top_p / speed are echoed into the run's parameters

Usage:  python scripts/params_test.py [ws://127.0.0.1:8000/ws]
"""

from __future__ import annotations

import asyncio
import json
import sys

import websockets

WS_URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8000/ws"


def check(cond: bool, label: str) -> None:
    if not cond:
        raise AssertionError(label)
    print(f"  ok: {label}")


async def run(ws, payload: dict) -> dict:
    """Send one generate request, collect the run into a summary dict."""
    await ws.send(json.dumps({"type": "generate", "speed": 0, **payload}))
    out = {"sampled": [], "steps": 0}
    while True:
        ev = json.loads(await ws.recv())
        if ev["type"] == "error":
            raise AssertionError(f"backend error: {ev['message']}")
        elif ev["type"] == "generation_start":
            out["params"] = ev["params"]
            out["template"] = ev["template"]
            out["prompt_tokens"] = ev["prompt_tokens"]
        elif ev["type"] == "sampled":
            out["sampled"].append({"id": ev["id"], "rank": ev["rank"]})
        elif ev["type"] == "generation_end":
            out["steps"] = ev["steps"]
            out["text"] = ev["text"]
            out["reason"] = ev["reason"]
            return out


async def main() -> None:
    async with websockets.connect(WS_URL, max_size=64 * 1024 * 1024) as ws:
        info = json.loads(await ws.recv())
        check(info["type"] == "model_info", "connected (model_info received)")

        print("\n[max_new_tokens]")
        r = await run(ws, {"prompt": "Explain quantum computing", "max_new_tokens": 5})
        check(r["steps"] <= 5, f"generated {r['steps']} tokens (limit 5)")
        check(r["steps"] == 5 or r["reason"] == "eos", "stopped for the right reason")

        print("\n[temperature = 0 -> greedy]")
        r = await run(ws, {"prompt": "The capital of France is", "max_new_tokens": 6, "temperature": 0})
        check(all(s["rank"] == 1 for s in r["sampled"]), "every sampled token has rank 1")
        check("greedy" in r["params"]["strategy"], f"strategy reported: {r['params']['strategy']}")

        print("\n[top_k = 1 at high temperature -> still deterministic pick]")
        r = await run(ws, {"prompt": "Once upon a time", "max_new_tokens": 6, "temperature": 1.5, "top_k": 1})
        check(all(s["rank"] == 1 for s in r["sampled"]), "rank 1 despite T=1.5 (top-k filter works)")

        print("\n[fixed seed -> reproducible]")
        a = await run(ws, {"prompt": "Write a short poem", "max_new_tokens": 8, "temperature": 0.9, "seed": 42})
        b = await run(ws, {"prompt": "Write a short poem", "max_new_tokens": 8, "temperature": 0.9, "seed": 42})
        check([s["id"] for s in a["sampled"]] == [s["id"] for s in b["sampled"]],
              "same seed produced identical token ids twice")
        c = await run(ws, {"prompt": "Write a short poem", "max_new_tokens": 8, "temperature": 0.9, "seed": 43})
        differs = [s["id"] for s in a["sampled"]] != [s["id"] for s in c["sampled"]]
        print(f"  note: different seed produced {'different' if differs else 'identical'} tokens")

        print("\n[chat vs raw mode]")
        chat = await run(ws, {"prompt": "hello", "max_new_tokens": 2, "chat_mode": "auto"})
        raw = await run(ws, {"prompt": "hello", "max_new_tokens": 2, "chat_mode": "raw"})
        check(chat["template"] is True, f"chat mode applied the template ({chat['prompt_tokens']} prompt tokens)")
        check(raw["template"] is False, f"raw mode skipped it ({raw['prompt_tokens']} prompt tokens)")
        check(chat["prompt_tokens"] > raw["prompt_tokens"], "template adds the wrapper tokens")

        print("\n[echo of remaining params]")
        r = await run(ws, {"prompt": "hi", "max_new_tokens": 2, "top_p": 0.5, "repetition_penalty": 1.4, "top_k": 7})
        check(r["params"]["top_p"] == 0.5, "top_p echoed")
        check(r["params"]["repetition_penalty"] == 1.4, "repetition_penalty echoed")
        check(r["params"]["top_k"] == 7, "top_k echoed")
        check("rep=1.4" in r["params"]["strategy"], "strategy string includes the penalty")

        print("\nPARAMS TEST PASS")


if __name__ == "__main__":
    asyncio.run(asyncio.wait_for(main(), timeout=600))
