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


def base_delay(event: dict, n_layer: int) -> float:
    """The pacing delay that follows one event in the live stream.

    Single source of truth for both live generation (``step_events`` below) and
    replay (``main.replay_run``), so a recorded run replays with the exact same
    cadence the client's speed control multiplies. Events with no post-delay in
    live (generation_start, step_end, generation_end, model_info) return 0.
    """
    first = event.get("step", 0) == 0
    step_scale = 1.0 if first else 0.3
    layer_scale = min(1.0, 6.0 / max(n_layer, 1)) * step_scale
    delays = {
        "tokenize": (0.7 if first else 0.08) * step_scale,
        "embeddings": (0.9 if first else 0.1) * step_scale,
        "positional": (0.5 if first else 0.02) * step_scale,
        "layer_start": 0.1 * layer_scale,
        "attention": 0.45 * layer_scale,
        "ffn": 0.35 * layer_scale,
        "layer_end": 0.06 * layer_scale,
        "anomaly": 0.02 * layer_scale,
        "logits": (0.8 if first else 0.3) * step_scale,
        "sampled": (0.6 if first else 0.25) * step_scale,
    }
    return delays.get(event.get("type", ""), 0.0)


def step_events(trace: dict, step: int, n_layer: int) -> Iterator[Event]:
    def ev(event_type: str, payload: dict) -> Event:
        event = {"type": event_type, "step": step, **payload}
        return event, base_delay(event, n_layer)

    yield ev("tokenize", {"tokens": trace["tokens"], "seq_len": trace["seq_len"]})
    yield ev("embeddings", trace["embeddings"])
    yield ev("positional", trace["positional"])

    for layer in trace["layers"]:
        i = layer["layer"]
        yield ev("layer_start", {"layer": i})
        yield ev("attention", {"layer": i, **layer["attention"]})
        if layer["ffn"] is not None:
            yield ev("ffn", {"layer": i, **layer["ffn"]})
        yield ev(
            "layer_end",
            {
                "layer": i,
                "hidden_norm": layer["hidden_norm"],
                "residual_delta": layer["residual_delta"],
            },
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
        )

    yield ev("logits", trace["logits"])
    yield ev("sampled", trace["sampled"])
