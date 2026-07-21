# Changelog

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
