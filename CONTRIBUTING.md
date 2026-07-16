# Contributing to InsideAI

Thanks for your interest! The project's one hard rule: **no fake data**.
Every visual must be driven by tensors extracted from the real model; where
data is reduced (pooled, projected, quantized), the reduction is documented
in code and labeled in the UI.

## Dev setup

```bash
# backend
cd backend
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows
.venv/Scripts/python run.py                        # ws://127.0.0.1:8000/ws

# frontend
cd frontend
npm install
npm run dev                                        # http://localhost:3000
```

## Tests

- Protocol invariants (causal masks, stochastic rows, decode rows, wiring):
  `backend/.venv/Scripts/python backend/scripts/ws_smoke_test.py`
- Browser end-to-end (uses your installed Edge headlessly, both servers
  running): `cd frontend && npm run test:e2e`

Run both before opening a PR. The E2E test is timing-sensitive on loaded
machines; a fail whose `scripts/fail.png` shows a healthy mid-generation UI
is usually CPU contention, not a regression — re-run it.

## Pull requests

- Keep the honest-data rule: if you add a visual, say in the PR exactly which
  tensor drives it and what reduction is applied.
- New chart colors must pass a CVD/contrast check (see the palette notes in
  `frontend/src/lib/palette.ts`).
- Match the existing code style; TypeScript is strict, Python is typed where
  practical.

## Ideas that would make great contributions

- Adapter for more model families (Mistral/Gemma MLP hook paths)
- An "agent mode" that wraps the model in a real retrieval / tool-call loop
  and visualizes those stages honestly
- Head-clustering or induction-head detection overlays
- Localized UI strings
