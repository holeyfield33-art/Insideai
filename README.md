# InsideAI

**Watch a real AI model think.** InsideAI runs a genuine language model on
your machine and turns every step of its computation into a live, interactive
3D visualization — tokenization, embeddings, attention, neurons firing,
probabilities, sampling, and the answer streaming out. No videos, no papers,
no mock data: what you see is the model, running.

![Attention arc diagram](docs/screenshots/attention.png)

## Is there a real LLM behind this?

**Yes.** By default InsideAI runs **Qwen2.5-0.5B-Instruct** (494M parameters,
24 layers) locally in the Python backend — it actually answers your prompt.
Every visual is driven by tensors extracted from its live forward pass:

- attention arcs and heatmaps are the model's real `softmax(QKᵀ/√d)` matrices
- the neuron diagram lights up with actual GELU/SiLU activations, wired by
  the model's true weight matrices
- probability bars are the exact distribution the sampler draws from
- the layer stack glows by how much each layer really changed the token's
  meaning (‖Δresidual‖)

Where data must be reduced to fit a screen (4864 neurons → 128 groups, 896
dimensions → 3 PCA axes), the reduction is documented and labeled in the UI —
never invented. Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## What you'll see

| | |
|---|---|
| ![Neurons](docs/screenshots/neurons.png) | **Neurons** — the classic textbook network, live: green input → blue hidden neurons → red output. Wire brightness = real connection strength × the signal flowing through it, one pulse per generated token. |
| ![Embeddings](docs/screenshots/embeddings.png) | **Embeddings** — every token becomes a point in space (PCA view of the real vectors); similar meanings sit close together. |
| ![Layers](docs/screenshots/layers.png) | **Layers** — the transformer stack. Amber = computing right now; deeper blue = that layer changed the meaning more. |
| ![Pipeline](docs/screenshots/pipeline.png) | **Pipeline** — the whole journey with live examples: words → token ids → vectors → 24 layers → scores → one word chosen → text streaming out. |

Plus a per-head attention heatmap, top-10 probability panel with the sampled
token highlighted, a token stream (green = your prompt, red = generated), and
an **Auto Tour** that flies the camera through the pipeline once per
generation.

## Quick start

Requirements: **Python 3.11+**, **Node 20+**, ~2 GB disk for the model
(downloaded once from HuggingFace on first start).

**Terminal 1 — backend:**

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python run.py
# macOS/Linux:
#   .venv/bin/pip install -r requirements.txt && .venv/bin/python run.py
```

**Terminal 2 — frontend:**

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000**, type a prompt, press **Generate**.
Windows shortcut: `scripts\dev-all.ps1` starts both.

## Using it

| Control | What it does |
|---|---|
| **Generate / Stop** | run the model on your prompt / cancel mid-generation |
| **Overview · Embeddings · Attention · Layers · Neurons** | fly the camera to a zone (and stay there) |
| **✦ Auto Tour** | one guided camera pass through the pipeline per generation |
| **Attention panel** | pick any layer and head; hover the heatmap for exact query→key weights |
| **⚙ settings** | tokens, temperature, top-k, top-p, repetition penalty, Chat/Raw mode, pacing (Cinematic/Fast/Instant), seed |
| **Chat vs Raw** | Chat applies the model's chat template (it answers you); Raw continues your text |
| **seed** | fixed seed → identical run, great for demos |

## Configuration

Backend env vars (defaults in parentheses):

| Var | Purpose |
|---|---|
| `INSIDEAI_MODEL` (`Qwen/Qwen2.5-0.5B-Instruct`) | any GPT-2- or Llama-family causal LM — try `distilgpt2` (fastest), `gpt2`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `INSIDEAI_DTYPE` (`bfloat16`) | `float32` for exactness, bf16 halves RAM |
| `INSIDEAI_THREADS` (half your cores) | CPU threads for the model — leaves headroom for the browser |
| `INSIDEAI_PORT` (`8000`) · `INSIDEAI_DEVICE` (`cpu`) | server basics; `cuda` if you have a GPU PyTorch build |
| `INSIDEAI_MAX_PROMPT_TOKENS` (64) · `INSIDEAI_MAX_NEW_TOKENS` (96) | sequence budgets |

Frontend: `NEXT_PUBLIC_WS_URL` (`ws://127.0.0.1:8000/ws`).

## Testing

```bash
# protocol test — asserts properties only true of real transformer internals
# (causal masks, stochastic attention rows, decode rows, wiring shapes)
backend/.venv/Scripts/python backend/scripts/ws_smoke_test.py

# browser end-to-end (needs both servers; uses your Edge headlessly)
cd frontend && npm run test:e2e
```

## How it works (short version)

The backend loads the model with eager attention (`output_attentions=True`)
and runs **prefill + KV-cache decode** like a production inference engine:
the prompt yields full attention matrices once, then each new token streams
its attention rows, pooled MLP activations (captured with forward hooks),
hidden-state norms, top-10 logits and the sampling decision over a WebSocket.
The frontend batches events per animation frame into a zustand store; the 3D
scene and HUD read from it. Full details, protocol table and the list of
honest reductions: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```
Prompt → Tokenizer → Embeddings (+position) → [ attention + MLP ] × 24
      → Logits → Sampling → next token → … → Streaming answer
```

## Project structure

```
backend/
  app/config.py                env-tunable settings
  app/main.py                  FastAPI + /ws session handling
  app/model/engine.py          model loading, hooks, prefill/decode traces, sampling
  app/model/reduce.py          PCA / pooling / quantization (documented reductions)
  app/streaming/protocol.py    trace → ordered, paced event stream
  scripts/ws_smoke_test.py     protocol invariants test
frontend/
  src/lib/                     types (protocol mirror), store, ws client, validated palette
  src/components/hud/          pipeline, heatmap, probabilities, token stream, controls
  src/components/scene/        R3F: embeddings, attention arcs, layer tower, neuron diagram
  scripts/e2e.js               headless-browser end-to-end test
docs/ARCHITECTURE.md           deep dive
```

## Troubleshooting

- **First start is slow** — the model downloads once into the HF cache.
- **Port 8000 busy** — set `INSIDEAI_PORT` and `NEXT_PUBLIC_WS_URL`.
- **Slow generation** — close heavy apps (screen recorders, many browser
  tabs); the model shares your CPU. `INSIDEAI_MODEL=distilgpt2` is ~6× smaller.
- **`attentions` empty on a custom model** — keep `attn_implementation="eager"`.

## Contributing & license

PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). The one hard rule:
**no fake data.** [MIT](LICENSE).
