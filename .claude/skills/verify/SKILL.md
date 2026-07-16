---
name: verify
description: Build, launch, and drive InsideAI (FastAPI + Qwen/distilgpt2 backend, Next.js 3D frontend) to verify changes end-to-end.
---

# Verify InsideAI

## Launch

Backend (default model Qwen/Qwen2.5-0.5B-Instruct, bfloat16, loads in ~15-25s;
`INSIDEAI_MODEL=distilgpt2` for the lightest run):

```powershell
E:\insideai\backend\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir E:\insideai\backend --host 127.0.0.1 --port 8000
```

Frontend (prod: `npm run build` then `npm run start`; dev: `npm run dev`) in
`E:\insideai\frontend` → http://localhost:3000.

## Drive

1. **Protocol surface** (asserts real-model invariants — causal mask on prefill
   `mode:"full"` matrices, decode `mode:"row"` rows spanning the context,
   per-head maps, descending top-k):
   `backend\.venv\Scripts\python backend\scripts\ws_smoke_test.py`
2. **Engine-only staged test** (bypasses WS; pinpoints hangs with faulthandler
   and prints per-stage timings): pattern in git history /
   scratchpad `engine_direct_test.py` — load → encode → trace_prefill →
   trace_decode ×3.
3. **GUI surface**: `npm run test:e2e` in `frontend/` (playwright-core +
   system Edge headless, `--enable-unsafe-swiftshader`). Screenshots land in
   `frontend/scripts/`.

## Gotchas (earned the hard way)

- **This machine runs heavy ambient software** (Void editor with ~45 node
  processes, FocuSee screen recorder, antivirus, the claude CLI). E2E timing
  is load-sensitive: the same suite passes idle and times out under load —
  a failed run whose fail.png shows a healthy mid-generation UI with huge
  ms/step is contention, not a regression. The backend caps torch threads
  (`INSIDEAI_THREADS`, default half the cores) so browser + model share CPU.
- **Kill only your own node processes**: filter
  `Win32_Process CommandLine -match "insideai"` — plain `Get-Process node`
  includes the user's editor.

- **This machine is RAM-starved** (15.7GB, often <1-2GB free). fp32 0.5B swaps
  → I/O-bound "hangs" with ~0 CPU. Keep bfloat16 (`INSIDEAI_DTYPE`). Under
  test, headless Edge (SwiftShader) + backend compete for CPU/RAM: server
  steps can stretch from ~0.3s to ~4s. E2E uses 1280×720 + 24 tokens +
  180-240s timeouts for this reason.
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
