# InsideAI — Architecture

This document explains how InsideAI extracts real transformer internals and
turns them into a live visualization. The one rule everywhere: **values are
reduced (projected, pooled, quantized) but never invented.**

```
┌──────────────────────────────┐        ┌──────────────────────────────────┐
│  backend (Python/FastAPI)    │  WS    │  frontend (Next.js/R3F)          │
│                              │ ─────▶ │                                  │
│  Qwen2.5-0.5B-Instruct       │ events │  zustand store (event reducer)   │
│  (or any GPT-2/Llama-family) │        │   ├─ 3D scene (four zones)       │
│  eager attention + hooks     │        │   └─ HUD (pipeline, heatmap,     │
│  prefill + KV-cache decode   │        │       probabilities, tokens)     │
└──────────────────────────────┘        └──────────────────────────────────┘
```

## 1. Inference: prefill + decode (like real engines)

- **Prefill** — the whole prompt runs in one forward pass, warming the KV
  cache. Every layer yields its complete `seq × seq` attention matrix.
- **Decode** — each new token is one forward pass against the cache. Each
  layer yields that token's attention **row** (how it reads the entire
  context). The client appends rows to its accumulated matrix; because causal
  attention means earlier tokens can never attend to later ones, the ragged
  accumulated matrix is *exact* — missing cells are true zeros.

This is what makes the stream realtime: decode steps ship O(seq) data instead
of O(seq²), and compute per token is a single cached forward pass
(~0.2–0.5 s on a laptop CPU for the 0.5B default model).

## 2. What is extracted, and how

| Signal | Source | Notes |
|---|---|---|
| Tokens | tokenizer | raw BPE string + decoded text + id, prompt/generated flag |
| Chat template | `tokenizer.apply_chat_template` | instruct models answer the prompt; template tokens are visible in the UI |
| Embeddings | `model.get_input_embeddings()(ids)` | PCA (SVD) → top-3 components per token |
| Positional | GPT-2: learned `wpe` table · Llama-family: RoPE `inv_freq` angles | kind reported in `model_info` |
| Attention | `output_attentions=True` with `attn_implementation="eager"` | sdpa/flash never materialize the matrix — eager is required |
| MLP activations | forward hook on `mlp.act` (GELU, GPT-2) / `mlp.act_fn` (SiLU, Llama) | newest position's vector |
| MLP input/output | forward hook on the `mlp` module | pre-MLP hidden state and MLP output, pooled |
| MLP wiring | `|W|` block-means of `c_fc/c_proj` (GPT-2) or `gate_proj/down_proj` (Llama) | computed once at load, normalized per matrix |
| Residual stream | `hidden_states[i+1] − hidden_states[i]` (last token) | per-layer ‖Δh‖ drives the tower glow |
| Logits | `out.logits[0, −1]` | top-10 of the *actual* sampling distribution |
| Sampling | repetition penalty → temperature → top-k → nucleus → multinomial | seedable; the displayed distribution is exactly what sampling draws from |

## 3. Honest reductions

| Wire data | Reduction | What is preserved |
|---|---|---|
| Embedding points | d-dim → PCA top-3, cloud rescaled | relative geometry of the sequence's embeddings |
| Attention weights | float → 0..255 quantization (÷255 client-side) | 1/255 resolution — below what a color ramp can show |
| Attention shape | prefill: full matrices (per-head while seq ≤ 64); decode: per-token rows | accumulated causal matrix is exact |
| MLP activations | n_inner (4864/3072) → 128 pooled channels | which regions fire, and how hard |
| MLP wiring | weight matrices → 16×12 / 12×16 block-means | real relative connection strengths at diagram resolution |
| Probabilities | top-10 of the penalized, temperature-scaled softmax + entropy | the head of the actual sampling distribution |

Every reduction is labeled in the UI ("128 pooled channels of 4864", "each
circle is a group of real neurons", …).

## 4. WebSocket protocol

Client → server:

```json
{"type":"generate","prompt":"...","max_new_tokens":48,"temperature":0.8,
 "top_k":40,"top_p":0.95,"repetition_penalty":1.15,"chat_mode":"auto",
 "seed":null,"speed":0.4}
{"type":"stop"}  {"type":"ping"}
```

Server → client, per step (pacing delays only affect *when*, never *what*):

| Event | Payload |
|---|---|
| `model_info` | architecture, params, positional kind, **mlp_wiring** (on connect) |
| `generation_start` | echoed prompt + clamped params + template flag |
| `tokenize` | token views for the full sequence |
| `embeddings` | PCA-3D point + norm per token |
| `positional` | learned-table sample or RoPE cos-angle sample |
| `layer_start` / `layer_end` | layer index · hidden norm, ‖Δresidual‖ |
| `attention` | `mode:"full"` (prefill): mean + per-head matrices · `mode:"row"` (decode): per-head rows + entropy |
| `ffn` | 128 pooled activations, firing fraction, pooled MLP input/output |
| `logits` | top-10 with prob + raw logit, entropy (bits) |
| `sampled` | chosen token, prob, rank, strategy, is_eos |
| `step_end` | decoded text so far, compute ms |
| `generation_end` | final text, duration, tokens/s, stop reason |

## 5. Frontend data flow

- **Store** (`src/lib/store.ts`): one zustand store; every event lands in
  `applyEvent`. Decode attention rows are appended to the accumulated causal
  matrix per layer.
- **Frame batching** (`src/lib/ws.ts`): incoming events queue and flush once
  per animation frame (with a timeout fallback for hidden tabs) — one React
  render per frame even during a 24-layer decode burst.
- **Transient animation**: R3F `useFrame` loops read `useSimStore.getState()`
  directly, so 60 fps animation never re-renders React.
- **Stability rule**: the heatmap, attention arcs and neuron diagram show the
  *user-selected* layer (default: the last layer) and update once per token.
  Only the tower follows the live per-layer sweep (`activeLayer`) — that is
  where "layer by layer" is the story.

## 6. The four zones

| Zone | What it shows | Driven by |
|---|---|---|
| Embeddings (x=−26) | one outlined point per token, drop-lines to a floor grid, PC axes | PCA of real embedding vectors |
| Attention (x=0) | tokens in reading order + arc diagram above | accumulated attention matrix of the selected layer/head |
| Layers (x=18) | outlined slab per layer; amber = computing, blue depth = ‖Δresidual‖ | `layer_start/end` events |
| Neurons (x=42) | classic layered network: green input → blue hidden → red output; wire brightness = \|W\| × live source activation | wiring from `model_info`, activations from `ffn` events |

Colors are validated for color-vision deficiency with a six-checks validator
(identity pair ΔE ≥ 12 under protan/deutan simulation, ≥ 3:1 contrast on the
card surface, monotone-lightness sequential ramps).

## 7. Performance

- bfloat16 weights (`INSIDEAI_DTYPE`) — halves RAM; fp32 0.5B swaps on 16 GB
  machines and turns into a page-fault storm.
- `torch.set_num_threads(cores/2)` (`INSIDEAI_THREADS`) — the model and the
  browser share the CPU instead of starving each other.
- uint8 attention quantization, row streaming, one instanced/merged geometry
  per visual layer, DOM label thinning for long sequences.
