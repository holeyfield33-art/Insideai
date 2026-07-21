"""Slices one step trace into an ordered stream of websocket events.

Pacing controls only *when* an event is sent — never *what* it contains. Every
payload field is the engine's real trace data.

Delay design: the first step (prefill) plays the full pipeline ceremony once;
decode steps run much tighter so generation flows in real time. Per-layer
delays scale with 6/n_layer so a 24-layer model takes about as long per sweep
as a 6-layer one. The client's `speed` factor multiplies everything
(0 = instant, useful for tests and impatient humans).
"""

from __future__ import annotations

from typing import Iterator

Event = tuple[dict, float]  # (payload, delay-after in seconds)


def step_events(trace: dict, step: int, n_layer: int) -> Iterator[Event]:
    first = step == 0
    step_scale = 1.0 if first else 0.3
    layer_scale = min(1.0, 6.0 / max(n_layer, 1)) * step_scale

    def ev(event_type: str, payload: dict, delay: float) -> Event:
        return {"type": event_type, "step": step, **payload}, delay

    yield ev("tokenize", {"tokens": trace["tokens"], "seq_len": trace["seq_len"]}, (0.7 if first else 0.08) * step_scale)
    yield ev("embeddings", trace["embeddings"], (0.9 if first else 0.1) * step_scale)
    yield ev("positional", trace["positional"], (0.5 if first else 0.02) * step_scale)

    for layer in trace["layers"]:
        i = layer["layer"]
        yield ev("layer_start", {"layer": i}, 0.1 * layer_scale)
        yield ev("attention", {"layer": i, **layer["attention"]}, 0.45 * layer_scale)
        if layer["ffn"] is not None:
            yield ev("ffn", {"layer": i, **layer["ffn"]}, 0.35 * layer_scale)
        yield ev(
            "layer_end",
            {
                "layer": i,
                "hidden_norm": layer["hidden_norm"],
                "residual_delta": layer["residual_delta"],
            },
            0.06 * layer_scale,
        )
        # zeta_proxy/flagged are unitarity-lab's real passive-mode telemetry
        # (engine.py's PassiveTelemetryHook): zeta_raw from the model's own
        # activations, flagged from VAR's calibrated SpectralRuptureDetector
        # — not synthesized here.
        yield ev(
            "anomaly",
            {
                "layer": i,
                "zeta_proxy": layer["zeta_proxy"],
                "flagged": layer["flagged"],
                "source": "unitarity-lab",
            },
            0.02 * layer_scale,
        )

    yield ev("logits", trace["logits"], (0.8 if first else 0.3) * step_scale)
    yield ev("sampled", trace["sampled"], (0.6 if first else 0.25) * step_scale)
