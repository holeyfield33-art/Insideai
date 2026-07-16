"""TransformerEngine — loads a real HuggingFace causal LM and traces every
internal the visualization needs, using the same prefill/decode architecture
as production inference engines:

  * prefill  — one forward pass over the whole prompt (KV cache warm-up).
    Yields complete (seq x seq) attention matrices for every layer.
  * decode   — one forward pass per new token against the KV cache.
    Yields each layer's attention *row* for the new token (how it reads the
    entire context), which the client accumulates into the full causal
    matrix. O(seq) per step instead of O(seq^2) — this is what makes the
    stream feel realtime.

Nothing in a trace is synthesized: tokens come from the tokenizer, embedding
coordinates are a PCA projection of the actual embedding table lookups,
attention matrices are the model's softmax(QK^T/sqrt(d)) outputs (eager
attention), neuron activations are captured with forward hooks on each MLP's
activation function, and probabilities come from the model's logits after the
exact temperature / repetition-penalty transform used for sampling.

Supported families:
  * gpt2  (distilgpt2, gpt2, gpt2-medium…): blocks at transformer.h,
    GELU hook at mlp.act, learned positional table wpe.
  * llama (Qwen2.x, Llama, Mistral…): blocks at model.layers, SiLU hook at
    mlp.act_fn, rotary position embeddings (RoPE).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from ..config import settings
from . import reduce


@dataclass(frozen=True)
class GenerationParams:
    max_new_tokens: int = 48
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    repetition_penalty: float = 1.15
    seed: int | None = None

    @classmethod
    def from_payload(cls, payload: dict) -> "GenerationParams":
        def clamp(key: str, default, lo, hi, cast):
            try:
                value = cast(payload.get(key, default))
            except (TypeError, ValueError):
                value = default
            return max(lo, min(hi, value))

        seed_raw = payload.get("seed")
        try:
            seed = int(seed_raw) if seed_raw not in (None, "") else None
        except (TypeError, ValueError):
            seed = None

        return cls(
            max_new_tokens=clamp("max_new_tokens", 48, 1, settings.max_new_tokens_cap, int),
            temperature=clamp("temperature", 0.8, 0.0, 2.0, float),
            top_k=clamp("top_k", 40, 0, 200, int),
            top_p=clamp("top_p", 0.95, 0.05, 1.0, float),
            repetition_penalty=clamp("repetition_penalty", 1.15, 1.0, 2.0, float),
            seed=seed,
        )

    def describe(self) -> str:
        rep = f", rep={self.repetition_penalty:g}" if self.repetition_penalty > 1 else ""
        if self.temperature <= 0:
            return f"greedy(argmax{rep})"
        k = self.top_k if self.top_k > 0 else "off"
        return f"sample(T={self.temperature:g}, top_k={k}, top_p={self.top_p:g}{rep})"


class TransformerEngine:
    def __init__(self, model_name: str = settings.model_name, device: str = settings.device):
        self.model_name = model_name
        self.device = torch.device(device)
        self.loaded = False
        self.load_seconds = 0.0
        self.tokenizer = None
        self.model = None
        self.family = "unknown"
        # Written by MLP forward hooks during a forward pass: layer -> (n_inner,)
        self._mlp_last: dict[int, torch.Tensor] = {}
        # layer -> (pre-MLP hidden, MLP output) of the newest position.
        self._mlp_io: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # Pooled |W| wiring per layer, computed once at load.
        self._mlp_wiring: list[dict] = []
        # Forward passes share the hook buffers; serialize callers.
        self._lock = threading.Lock()

    # ------------------------------------------------------------- loading

    def load(self) -> None:
        t0 = time.time()
        torch.set_num_threads(settings.torch_threads)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
            settings.dtype, torch.float32
        )
        # eager attention is required: sdpa/flash kernels never materialize the
        # attention matrix, so output_attentions would come back empty.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            output_attentions=True,
            output_hidden_states=True,
            attn_implementation="eager",
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.family = self._detect_family()
        self._register_mlp_hooks()
        self._mlp_wiring = self._compute_mlp_wiring()
        self.load_seconds = round(time.time() - t0, 2)
        self.loaded = True

    def _detect_family(self) -> str:
        transformer = getattr(self.model, "transformer", None)
        if transformer is not None and hasattr(transformer, "h"):
            return "gpt2"
        inner = getattr(self.model, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return "llama"
        return "unknown"

    def _blocks(self) -> list:
        if self.family == "gpt2":
            return list(self.model.transformer.h)
        if self.family == "llama":
            return list(self.model.model.layers)
        return []

    def _register_mlp_hooks(self) -> None:
        for i, block in enumerate(self._blocks()):
            mlp = getattr(block, "mlp", None)
            if mlp is None:
                continue

            # GPT-2 GELU lives at mlp.act; llama-family SiLU at mlp.act_fn.
            act = getattr(mlp, "act", None) or getattr(mlp, "act_fn", None)
            if act is not None:

                def act_hook(_module, _inputs, output, layer=i):
                    # (batch, seq, n_inner) -> the newest position's activations
                    self._mlp_last[layer] = output[0, -1].detach().clone()

                act.register_forward_hook(act_hook)

            def io_hook(_module, inputs, output, layer=i):
                # What flows in and out of the whole MLP for the newest token.
                self._mlp_io[layer] = (
                    inputs[0][0, -1].detach().float(),
                    output[0, -1].detach().float(),
                )

            mlp.register_forward_hook(io_hook)

    def _compute_mlp_wiring(self) -> list[dict]:
        """Per-layer pooled |W| block-means: the model's real connection
        strengths at diagram resolution (hidden x in, out x hidden)."""
        wiring = []
        n_in = settings.mlp_in_nodes
        n_hid = settings.mlp_hidden_nodes
        for block in self._blocks():
            mlp = getattr(block, "mlp", None)
            if mlp is None:
                continue
            if self.family == "gpt2":
                # Conv1D weights are (in_features, out_features).
                w_in = mlp.c_fc.weight.T  # (n_inner, n_embd)
                w_out = mlp.c_proj.weight.T  # (n_embd, n_inner)
            else:
                w_in = mlp.gate_proj.weight  # (n_inner, n_embd)
                w_out = mlp.down_proj.weight  # (n_embd, n_inner)
            wiring.append(
                {
                    "w_in": reduce.block_mean_2d(w_in.abs(), n_hid, n_in),
                    "w_out": reduce.block_mean_2d(w_out.abs(), n_in, n_hid),
                }
            )
        return wiring

    # ------------------------------------------------------------ metadata

    def _cfg(self, *names: str, default: int = 0) -> int:
        for name in names:
            value = getattr(self.model.config, name, None)
            if value:
                return int(value)
        return default

    @property
    def n_layer(self) -> int:
        return self._cfg("n_layer", "num_hidden_layers")

    @property
    def n_positions(self) -> int:
        return self._cfg("n_positions", "max_position_embeddings", default=1024)

    def model_info(self) -> dict:
        n_embd = self._cfg("n_embd", "hidden_size")
        n_inner = self._cfg("n_inner", "intermediate_size", default=4 * n_embd)
        param_count = sum(p.numel() for p in self.model.parameters())
        return {
            "model": self.model_name,
            "family": self.family,
            "chat_template": bool(getattr(self.tokenizer, "chat_template", None)),
            "n_layer": self.n_layer,
            "n_head": self._cfg("n_head", "num_attention_heads"),
            "n_embd": n_embd,
            "n_inner": n_inner,
            "n_positions": self.n_positions,
            "vocab_size": self._cfg("vocab_size"),
            "param_count": param_count,
            "param_count_h": reduce.fmt_params(param_count),
            "neuron_channels": settings.neuron_channels,
            "neurons_per_channel": max(1, n_inner // settings.neuron_channels),
            "mlp_wiring": {
                "in_nodes": settings.mlp_in_nodes,
                "hidden_nodes": settings.mlp_hidden_nodes,
                "layers": self._mlp_wiring,
            },
            "full_attention_seq_cap": settings.full_attention_seq_cap,
            "max_prompt_tokens": settings.max_prompt_tokens,
            "max_new_tokens_cap": settings.max_new_tokens_cap,
            "positional_kind": "learned (wpe)" if self.family == "gpt2" else "rotary (RoPE)",
            "device": str(self.device),
            "load_seconds": self.load_seconds,
        }

    @property
    def eos_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    def make_generator(self, seed: int | None) -> torch.Generator:
        gen = torch.Generator(device=self.device.type)
        gen.manual_seed(seed if seed is not None else int(time.time_ns() % (2**31)))
        return gen

    # ------------------------------------------------------------ encoding

    def encode_prompt(self, prompt: str, chat_mode: str = "auto") -> tuple[torch.Tensor, bool, bool]:
        """Prompt -> (1, n) input ids.

        Returns (ids, truncated, templated). In "auto" mode instruct models
        get their chat template applied, so the model actually answers the
        prompt instead of just continuing it.
        """
        content_ids = self.tokenizer.encode(prompt)
        truncated = len(content_ids) > settings.max_prompt_tokens
        if truncated:
            content_ids = content_ids[: settings.max_prompt_tokens]
            prompt = self.tokenizer.decode(content_ids)

        use_template = chat_mode != "raw" and bool(getattr(self.tokenizer, "chat_template", None))
        if use_template:
            templated = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
            # Depending on the transformers version this is a Tensor or a
            # BatchEncoding wrapping one.
            ids = templated if torch.is_tensor(templated) else templated["input_ids"]
        else:
            ids = torch.tensor([content_ids], dtype=torch.long)
        return ids.to(self.device), truncated, use_template

    def token_view(self, token_id: int, index: int, kind: str) -> dict:
        raw = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return {
            "index": index,
            "id": int(token_id),
            "raw": raw,
            "text": self.tokenizer.decode([token_id]),
            "kind": kind,  # "prompt" | "generated"
        }

    # ------------------------------------------------ shared trace pieces

    def _token_views(self, full_ids: torch.Tensor, prompt_len: int) -> list[dict]:
        return [
            self.token_view(tid, i, "prompt" if i < prompt_len else "generated")
            for i, tid in enumerate(full_ids[0].tolist())
        ]

    @torch.no_grad()
    def _embedding_snapshot(self, full_ids: torch.Tensor) -> dict:
        emb = self.model.get_input_embeddings()(full_ids)[0]  # (seq, d)
        return {
            "points": reduce.pca_3d(emb),
            "norms": reduce.norms(emb, 2),
            "dim": int(emb.shape[1]),
            "method": "pca3(token embedding table)",
        }

    @torch.no_grad()
    def _positional_snapshot(self, seq: int) -> dict:
        if self.family == "gpt2":
            pos_emb = self.model.transformer.wpe(torch.arange(seq, device=self.device))
            return {
                "kind": "learned",
                "dims_shown": settings.positional_dims,
                "norms": reduce.norms(pos_emb, 2),
                "sample": reduce.round_floats(pos_emb[:, : settings.positional_dims], 3),
            }
        rope = getattr(getattr(self.model, "model", None), "rotary_emb", None)
        inv_freq = getattr(rope, "inv_freq", None)
        if inv_freq is None:
            return {"kind": "none", "dims_shown": 0, "norms": [], "sample": []}
        dims = min(12, inv_freq.shape[0])
        angles = torch.arange(seq, device=inv_freq.device).float()[:, None] * inv_freq[None, :dims]
        return {
            "kind": "rotary",
            "dims_shown": int(dims),
            "norms": [1.0] * seq,  # RoPE rotates; it never changes vector norms
            "sample": reduce.round_floats(torch.cos(angles), 3),
        }

    def _extract_layers(self, attentions, hidden_states, full: bool) -> list[dict]:
        layers = []
        k_len = attentions[0].shape[-1]
        send_full_heads = k_len <= settings.full_attention_seq_cap
        for i, attn_batch in enumerate(attentions):
            attn = attn_batch[0].float()  # (n_head, q_len, k_len)
            rows = attn[:, -1, :]  # newest query position: (n_head, k_len)
            prev = hidden_states[i][0, -1].float()
            curr = hidden_states[i + 1][0, -1].float()

            if full:
                attention = {
                    "mode": "full",
                    "mean": reduce.quant255(attn.mean(dim=0)),
                    "heads": reduce.quant255(attn) if send_full_heads else None,
                    "last_row": reduce.quant255(rows),
                    "head_entropy_bits": reduce.row_entropy_bits(rows),
                    "quantization": 255,
                }
            else:
                attention = {
                    "mode": "row",
                    "mean_row": reduce.quant255(rows.mean(dim=0)),
                    "head_rows": reduce.quant255(rows),
                    "head_entropy_bits": reduce.row_entropy_bits(rows),
                    "quantization": 255,
                }

            mlp_vec = self._mlp_last.get(i)
            if mlp_vec is not None:
                pooled = reduce.pool_channels(mlp_vec, settings.neuron_channels)
                ffn = {
                    "activations": reduce.round_floats(pooled, 3),
                    "pooled_from": int(mlp_vec.shape[0]),
                    "mean_abs": round(float(mlp_vec.abs().mean().item()), 4),
                    "max": round(float(mlp_vec.max().item()), 3),
                    "active_frac": round(float((mlp_vec > 0).float().mean().item()), 3),
                }
                io = self._mlp_io.get(i)
                if io is not None:
                    x_in, x_out = io
                    ffn["input_pooled"] = reduce.round_floats(
                        reduce.pool_channels(x_in, settings.mlp_in_nodes), 3
                    )
                    ffn["output_pooled"] = reduce.round_floats(
                        reduce.pool_channels(x_out, settings.mlp_in_nodes), 3
                    )
            else:
                ffn = None

            layers.append(
                {
                    "layer": i,
                    "attention": attention,
                    "ffn": ffn,
                    "hidden_norm": round(float(curr.norm().item()), 2),
                    "residual_delta": round(float((curr - prev).norm().item()), 2),
                }
            )
        return layers

    def _repetition_penalty(self, logits: torch.Tensor, full_ids: torch.Tensor, penalty: float) -> torch.Tensor:
        """HF-style penalty: seen tokens' positive logits are divided by the
        penalty, negative ones multiplied — discourages loops honestly, and
        the displayed distribution reflects it."""
        if penalty <= 1.0:
            return logits
        penalized = logits.clone()
        seen = torch.unique(full_ids[0])
        values = penalized[seen]
        penalized[seen] = torch.where(values > 0, values / penalty, values * penalty)
        return penalized

    def _next_token(self, raw_logits: torch.Tensor, full_ids: torch.Tensor, params: GenerationParams, generator: torch.Generator) -> tuple[dict, dict]:
        """(logits payload, sampled payload) from the real next-token scores."""
        logits = self._repetition_penalty(raw_logits.float(), full_ids, params.repetition_penalty)
        temp = params.temperature
        scaled = logits / temp if temp > 0 else logits
        display = F.softmax(scaled, dim=-1)

        top_probs, top_ids = display.topk(settings.topk_probs)
        topk = [
            {
                "id": int(tid),
                "token": self.tokenizer.decode([int(tid)]),
                "raw": self.tokenizer.convert_ids_to_tokens([int(tid)])[0],
                "prob": round(float(p), 6),
                "logit": round(float(raw_logits[tid]), 3),
            }
            for p, tid in zip(top_probs.tolist(), top_ids.tolist())
        ]

        chosen = self._sample(logits, scaled, params, generator)
        rank = int((display > display[chosen]).sum().item()) + 1

        logits_payload = {
            "topk": topk,
            "entropy_bits": round(reduce.entropy_bits(display), 3),
            "temperature_used": temp,
            "vocab_size": int(display.shape[0]),
        }
        sampled_payload = {
            "id": chosen,
            "token": self.tokenizer.decode([chosen]),
            "raw": self.tokenizer.convert_ids_to_tokens([chosen])[0],
            "prob": round(float(display[chosen]), 6),
            "rank": rank,
            "strategy": params.describe(),
            "is_eos": chosen == self.eos_id,
        }
        return logits_payload, sampled_payload

    def _sample(self, logits: torch.Tensor, scaled: torch.Tensor, params: GenerationParams, generator: torch.Generator) -> int:
        """Standard temperature / top-k / nucleus sampling over real logits."""
        if params.temperature <= 0:
            return int(torch.argmax(logits).item())

        filtered = scaled.clone()
        if params.top_k > 0:
            kth = torch.topk(filtered, min(params.top_k, filtered.shape[-1])).values[-1]
            filtered[filtered < kth] = float("-inf")
        if params.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(filtered, descending=True)
            cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumulative > params.top_p
            remove[1:] = remove[:-1].clone()  # always keep the top token
            remove[0] = False
            filtered[sorted_idx[remove]] = float("-inf")

        probs = F.softmax(filtered, dim=-1)
        return int(torch.multinomial(probs, 1, generator=generator).item())

    # ----------------------------------------------------------- the trace

    def _assemble(self, mode: str, full_ids: torch.Tensor, prompt_len: int, out, params, generator, compute_ms: float) -> dict:
        seq = int(full_ids.shape[1])
        logits_payload, sampled_payload = self._next_token(
            out.logits[0, -1], full_ids, params, generator
        )
        return {
            "mode": mode,
            "seq_len": seq,
            "compute_ms": compute_ms,
            "tokens": self._token_views(full_ids, prompt_len),
            "embeddings": self._embedding_snapshot(full_ids),
            "positional": self._positional_snapshot(seq),
            "layers": self._extract_layers(out.attentions, out.hidden_states, full=(mode == "prefill")),
            "logits": logits_payload,
            "sampled": sampled_payload,
        }

    @torch.no_grad()
    def trace_prefill(self, input_ids: torch.Tensor, params: GenerationParams, generator: torch.Generator):
        """Process the whole prompt once. Full attention matrices + KV cache."""
        with self._lock:
            t0 = time.time()
            self._mlp_last.clear()
            self._mlp_io.clear()
            out = self.model(input_ids, use_cache=True, past_key_values=DynamicCache())
            compute_ms = round((time.time() - t0) * 1000.0, 1)
            trace = self._assemble(
                "prefill", input_ids, int(input_ids.shape[1]), out, params, generator, compute_ms
            )
        return trace, out.past_key_values

    @torch.no_grad()
    def trace_decode(self, full_ids: torch.Tensor, cache, params: GenerationParams, generator: torch.Generator, prompt_len: int):
        """One cached decode step for the newest token (the last id of full_ids)."""
        with self._lock:
            t0 = time.time()
            self._mlp_last.clear()
            self._mlp_io.clear()
            out = self.model(full_ids[:, -1:], use_cache=True, past_key_values=cache)
            compute_ms = round((time.time() - t0) * 1000.0, 1)
            trace = self._assemble(
                "decode", full_ids, prompt_len, out, params, generator, compute_ms
            )
        return trace, out.past_key_values


engine = TransformerEngine()
