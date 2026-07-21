---
name: verify
description: Build, launch, and drive InsideAI (FastAPI + Qwen/distilgpt2 backend, Next.js 3D frontend) to verify changes end-to-end.
---

# Verify InsideAI

## Launch

Backend (default model Qwen/Qwen2.5-0.5B-Instruct, bfloat16, loads in ~15-25s;
`INSIDEAI_MODEL=distilgpt2` for the lightest run — loads in a couple seconds
and is what this repo's telemetry pipe has actually been smoke-tested against):

Linux / macOS / Codespaces, from the repo root:

```bash
cd backend
python3 -m venv .venv          # first time only
.venv/bin/pip install -r requirements.txt
INSIDEAI_MODEL=distilgpt2 .venv/bin/python -m uvicorn app.main:app --app-dir . --host 127.0.0.1 --port 8000
```

(Verified: this exact sequence loads `distilgpt2` and serves `/ws` on a
fresh Linux container — see `CHANGELOG.md` for the run this is based on.)

Windows, absolute paths (original dev box):

```powershell
E:\insideai\backend\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir E:\insideai\backend --host 127.0.0.1 --port 8000
```

Frontend (prod: `npm run build` then `npm run start`; dev: `npm run dev`) in
`frontend/` → http://localhost:3000.

## Drive

1. **Protocol surface** (asserts real-model invariants — causal mask on prefill
   `mode:"full"` matrices, decode `mode:"row"` rows spanning the context,
   per-head maps, descending top-k, one `anomaly` event per layer per step):
   `backend/.venv/bin/python backend/scripts/ws_smoke_test.py`
   (`backend\.venv\Scripts\python` on Windows). Needs the `websockets`
   package in the venv (`pip install websockets`) and a running backend.
2. **Engine-only staged test** (bypasses WS; pinpoints hangs with faulthandler
   and prints per-stage timings): pattern in git history /
   scratchpad `engine_direct_test.py` — load → encode → trace_prefill →
   trace_decode ×3.
3. **GUI surface**: `npm run test:e2e` in `frontend/` (playwright-core,
   `channel: "msedge"` — needs Microsoft Edge installed). This only runs
   where Edge is present (the original Windows dev box); a bare Linux/
   Codespaces container ships Chromium, not Edge, so this test does not run
   there out of the box.

## Gotchas (earned the hard way)

- **Windows dev box specifics** (do not assume these on Linux/Codespaces):
  that machine ran heavy ambient software (Void editor with ~45 node
  processes, FocuSee screen recorder, antivirus, the claude CLI), so E2E
  timing was load-sensitive there — a failed run whose fail.png shows a
  healthy mid-generation UI with huge ms/step was contention, not a
  regression. It was also RAM-starved (15.7GB, often <1-2GB free): fp32 0.5B
  swapped → I/O-bound "hangs" with ~0 CPU, and headless Edge (SwiftShader) +
  backend competed for CPU/RAM (server steps stretched from ~0.3s to ~4s,
  hence 1280×720 + 24 tokens + 180-240s timeouts in the E2E suite). Keep
  bfloat16 (`INSIDEAI_DTYPE`) regardless of platform — it halves RAM either way.
- **On Windows**, kill only your own node processes: filter
  `Win32_Process CommandLine -match "insideai"` — plain `Get-Process node`
  includes the user's editor. On Linux, filter `ps aux | grep insideai`
  similarly before killing anything broader.
- The backend caps torch threads (`INSIDEAI_THREADS`, default half the
  cores) so browser + model share CPU — applies on every platform.
- **Playwright `waitForFunction(fn, options)` is WRONG** — options is the 3rd
  arg (`waitForFunction(fn, undefined, options)`); otherwise you silently get
  the 30s default.
- Selector `button:has-text('Cinematic')` is ambiguous (TopBar "✦ Cinematic"
  vs pacing segment): use `getByRole("button", { name: "Cinematic", exact: true })`.
- `apply_chat_template(return_tensors="pt")` returns a BatchEncoding in this
  transformers version — unwrap `["input_ids"]` (handled in engine).
- Generation runs entirely inside run_generation's `try` — if the client sees
  silence, check the backend log; a silently dead task means something escaped
  the try (regression).
- Something external polls `GET /v1/models` on :8000 (~1/s) — not ours;
  `INSIDEAI_PORT` moves the backend.
- Generation is nondeterministic unless `seed` is set.
