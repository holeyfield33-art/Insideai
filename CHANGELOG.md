# Changelog

## Phase 4 — record & replay runs

- **Recording (opt-in, default OFF).** `INSIDEAI_RECORD=1` makes the backend
  capture every WebSocket event of a generation, in emission order, to
  `results/runs/<ISO-date>_<insideai-sha>_<env>/run.jsonl` (one JSON event per
  line) plus a `manifest.json` reproducibility record (prompt, model, seed +
  the *effective* seed even for "random" runs, sampling params, git SHAs of all
  three repos — insideai/unitarity-lab/VAR — resolved python/torch/transformers/
  node versions, timestamp, total steps, generated text). Recording is a passive
  tap on the exact dicts already sent (`app/recording.py`); with it off, live
  behavior is byte-for-byte unchanged.
- **Replay over the same protocol.** New `GET /runs` lists recordings
  (`[{id, prompt, model, timestamp, steps}]`); a new `{"type":"replay","id"}`
  WS message re-emits a run's events verbatim, honoring the client's speed
  control via a shared `base_delay` (now the single source of truth for both
  live pacing and replay, in `protocol.py`). Replayed events are byte-identical
  to live ones, so the EKG panel, attention view and token stream render with no
  special-case paths — and zeta values are the recorded values exactly, never
  recomputed.
- **No model needed to replay.** `INSIDEAI_SKIP_MODEL=1` starts the backend in
  replay-only mode (`lifespan` skips `engine.load()`); the connect handshake
  sends `{"type":"server_mode","live":false}` and no `model_info` (the recorded
  `model_info` is the first replayed event instead). Live `generate` is refused
  with a clear error in this mode.
- **UI.** A `RunPicker` panel lists recorded runs by prompt + timestamp and
  replays the selected one; an unmistakable, always-visible `ReplayBanner`
  ("● REPLAY — recorded run") shows whenever a recording is on screen so it can
  never be mistaken for a live run (honesty requirement).
- Verified: `tsc --noEmit` + `next build` clean; the torch-free replay/listing/
  path-traversal-guard logic unit-tested; recorded→replay round-trip asserts
  byte-identical event streams and exact zeta equality against a model-less
  backend.

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
