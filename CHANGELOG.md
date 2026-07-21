# Changelog

## v1-live-zeta — Phase 3 checkpoint (swap placeholder for real telemetry)

- Added `unitarity-labs` to `backend/requirements.txt`, pinned to its
  Phase 2 checkpoint commit (`1dfe4e5`, `v4.0.0-slim`).
- `engine.py`: `TransformerEngine.load()` now wraps `self.model` in a
  passive-mode `UniversalHookWrapper` and builds a `PassiveTelemetryHook`
  once at load time (its hooks fire on every `self.model(...)` forward
  pass already made by `trace_prefill`/`trace_decode`, so no other call
  site changes). `_extract_layers` reads real `{zeta_raw, flagged}`
  once per forward pass — the hook reports one cross-layer coherence
  signal per step, so every layer in that step's sweep shares the same
  real reading rather than a distinct-per-layer placeholder formula.
- `protocol.py`: the `anomaly` event's `flagged` now comes from
  `layer["flagged"]` (VAR's real `SpectralRuptureDetector`) instead of a
  hardcoded `False`; `source` is `"unitarity-lab"`.
- Updated the two stale "PLACEHOLDER" comments left in the frontend
  (`AnomalyEkgPanel.tsx`, `types.ts`) — no frontend logic changed.
- Verified live against a running backend (`distilgpt2`,
  `scripts/ws_smoke_test.py` — PASS, 18 real `anomaly` events across 3
  steps) and a direct WS probe: `zeta_raw` is real and changes step to
  step (e.g. `0.997 → 0.916 → 0.920 → 0.916` across 4 generated tokens),
  not the old static `residual_delta / hidden_norm` ratio.
- `tsc --noEmit` and `next build` both pass; diff touches only
  `requirements.txt`, `engine.py`, `protocol.py`, and the two
  comment-only frontend files above.

## v0-fork-baseline — Phase 0 checkpoint (fork & setup)

- Applied `insideai-anomaly-pipe.patch`: adds a placeholder `zeta_proxy`
  metric (`residual_delta / hidden_norm`, computed from real trace data,
  not synthetic) to each layer in `engine.py`, a placeholder `anomaly` WS
  event (`flagged` hardcoded `false` — no detector runs yet) in
  `protocol.py`, and a frontend EKG panel (`AnomalyEkgPanel.tsx`) that
  visualizes the pipe end-to-end. This proves the anomaly-event pipe
  works before Phase 3 swaps `zeta_proxy` for the real unitarity-lab
  passive-mode hook.
- Verified: `python3 -c "from app.model.engine import engine; from
  app.streaming.protocol import step_events"` imports clean, `npm run
  build` succeeds, `zeta_proxy` appears in `engine.py`.

Note: this repo's checkpoint is recorded here instead of as a pushed git
tag — this session's git proxy rejects tag-ref pushes (HTTP 403), so
`v0-fork-baseline` is a checkpoint marker, not an actual git tag. Push
`git tag v0-fork-baseline` from a local clone if a real tag is wanted.
