"""End-to-end smoke test for the InsideAI websocket stream.

Connects to a running backend, generates a few tokens at speed 0, and asserts
that every event type arrives with real, shape-consistent model data —
including checks that only hold for genuine transformer outputs (causal
attention masks, stochastic-matrix rows, descending top-k probabilities).

Usage:  python scripts/ws_smoke_test.py [ws://127.0.0.1:8000/ws]
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter

import websockets

WS_URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8000/ws"
PROMPT = "Explain quantum computing"
STEPS = 3

REQUIRED_EVENTS = {
    "tokenize",
    "embeddings",
    "positional",
    "layer_start",
    "attention",
    "ffn",
    "layer_end",
    "logits",
    "sampled",
    "step_end",
    "generation_end",
}


def check(cond: bool, label: str) -> None:
    if not cond:
        raise AssertionError(label)
    print(f"  ok: {label}")


async def main() -> None:
    async with websockets.connect(WS_URL, max_size=32 * 1024 * 1024) as ws:
        info = json.loads(await ws.recv())
        check(info["type"] == "model_info", "first frame is model_info")
        n_layer, n_head = info["n_layer"], info["n_head"]
        neuron_channels = info["neuron_channels"]
        wiring = info["mlp_wiring"]
        check(len(wiring["layers"]) == n_layer, "model_info: MLP wiring for every layer")
        check(
            len(wiring["layers"][0]["w_in"]) == wiring["hidden_nodes"]
            and len(wiring["layers"][0]["w_in"][0]) == wiring["in_nodes"],
            "model_info: wiring matrices have diagram shape",
        )
        print(
            f"  model={info['model']} ({info['family']}, chat_template={info['chat_template']}) "
            f"layers={n_layer} heads={n_head} d_model={info['n_embd']} params={info['param_count_h']}"
        )

        await ws.send(
            json.dumps(
                {
                    "type": "generate",
                    "prompt": PROMPT,
                    "max_new_tokens": STEPS,
                    "temperature": 0.8,
                    "top_k": 40,
                    "top_p": 0.95,
                    "seed": 7,
                    "speed": 0,
                }
            )
        )

        counts: Counter = Counter()
        seq_len = 0
        row_checked = False
        final = None
        while True:
            ev = json.loads(await ws.recv())
            counts[ev["type"]] += 1

            if ev["type"] == "error":
                raise AssertionError(f"backend error: {ev['message']}")

            if ev["type"] == "tokenize":
                seq_len = ev["seq_len"]
                check(len(ev["tokens"]) == seq_len, f"tokenize: {seq_len} token views")

            elif ev["type"] == "embeddings" and counts["embeddings"] == 1:
                check(len(ev["points"]) == seq_len, "embeddings: one 3D point per token")
                check(
                    all(len(p) == 3 and all(-1.001 <= v <= 1.001 for v in p) for p in ev["points"]),
                    "embeddings: points are normalized 3-vectors",
                )

            elif ev["type"] == "attention" and counts["attention"] == 1:
                check(ev["mode"] == "full", "attention: prefill streams full matrices")
                mean = ev["mean"]
                check(
                    len(mean) == seq_len and all(len(r) == seq_len for r in mean),
                    "attention: mean matrix is seq x seq",
                )
                upper = [mean[i][j] for i in range(seq_len) for j in range(seq_len) if j > i]
                check(all(v == 0 for v in upper), "attention: causal mask (upper triangle is zero)")
                check(195 <= sum(mean[-1]) <= 320, "attention: last row sums to ~1.0 (255 quantized)")
                check(ev["heads"] is not None and len(ev["heads"]) == n_head, "attention: per-head matrices present")
                check(len(ev["head_entropy_bits"]) == n_head, "attention: per-head entropy")

            elif ev["type"] == "attention" and ev.get("mode") == "row" and not row_checked:
                row_checked = True
                check(len(ev["mean_row"]) == seq_len, "attention: decode row spans the full context")
                check(len(ev["head_rows"]) == n_head, "attention: per-head decode rows")
                check(195 <= sum(ev["mean_row"]) <= 320, "attention: decode row sums to ~1.0")

            elif ev["type"] == "ffn" and counts["ffn"] == 1:
                check(len(ev["activations"]) == neuron_channels, f"ffn: {neuron_channels} pooled channels")
                check(0.0 < ev["active_frac"] < 1.0, "ffn: plausible GELU active fraction")
                check(
                    len(ev.get("input_pooled", [])) == wiring["in_nodes"]
                    and len(ev.get("output_pooled", [])) == wiring["in_nodes"],
                    "ffn: pooled MLP input/output vectors present",
                )

            elif ev["type"] == "logits" and counts["logits"] == 1:
                probs = [t["prob"] for t in ev["topk"]]
                check(len(probs) == 10, "logits: top-10 candidates")
                check(all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1)), "logits: probs descending")
                check(0 < sum(probs) <= 1.0001, "logits: top-k mass is a valid probability mass")
                check(ev["entropy_bits"] > 0, "logits: positive entropy")

            elif ev["type"] == "sampled" and counts["sampled"] == 1:
                check(ev["rank"] >= 1 and ev["prob"] > 0, "sampled: rank/prob from real distribution")

            elif ev["type"] == "generation_end":
                final = ev
                break

        missing = REQUIRED_EVENTS - set(counts)
        check(not missing, f"all event types seen (missing: {missing or 'none'})")
        check(row_checked or final["steps"] < 2, "attention: decode-row events observed")
        check(counts["layer_start"] == n_layer * final["steps"], "layer events cover every layer x step")
        check(counts["step_end"] == final["steps"], "one step_end per generated token")
        check(len(final["text"]) > 0, "generated text is non-empty")

        print(f"\n  events: {dict(counts)}")
        print(f"  prompt:    {PROMPT!r}")
        print(f"  generated: {final['text']!r}  ({final['reason']}, {final['duration_s']}s)")
        print("\nSMOKE TEST PASS")


if __name__ == "__main__":
    asyncio.run(asyncio.wait_for(main(), timeout=180))
