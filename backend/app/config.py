"""Central configuration. Every knob can be overridden with an INSIDEAI_* env var."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


# Repo root = .../<repo>/backend/app/config.py -> parents[2]. Used so recorded
# runs land in <repo>/results/runs regardless of the process's working dir.
_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    # Any GPT-2- or Llama-family HuggingFace causal LM works. The default is a
    # small *instruct* model so answers are actually useful; set
    # INSIDEAI_MODEL=distilgpt2 for the fastest possible boot.
    model_name: str = os.getenv("INSIDEAI_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    # Load weights from a local GGUF file instead of downloading model_name
    # from the Hub — e.g. a model already pulled by Ollama or llama.cpp.
    # transformers dequantizes it into a real nn.Module at load time, so
    # every hook/introspection path below works unmodified. model_name is
    # still used as the display label. Only architectures transformers'
    # GGUF loader supports will work (gguf_file, not the containing repo,
    # decides that — see transformers.integrations.ggml.GGUF_CONFIG_MAPPING).
    gguf_file: str = os.getenv("INSIDEAI_GGUF_FILE", "").strip()
    device: str = os.getenv("INSIDEAI_DEVICE", "cpu")
    # bfloat16 halves RAM (a 0.5B model: ~2GB fp32 -> ~1GB). On RAM-starved
    # machines fp32 gets paged out and inference turns into a page-fault storm.
    dtype: str = os.getenv("INSIDEAI_DTYPE", "bfloat16")

    # Leave CPU headroom for the browser rendering the visualization —
    # torch grabbing every core starves the UI and vice versa.
    torch_threads: int = _env_int("INSIDEAI_THREADS", max(1, (os.cpu_count() or 4) // 2))

    host: str = os.getenv("INSIDEAI_HOST", "127.0.0.1")
    port: int = _env_int("INSIDEAI_PORT", 8000)

    # Sequence budgets. Prefill attention payloads grow O(seq^2); decode steps
    # are O(seq) rows, so generation length is cheap.
    max_prompt_tokens: int = _env_int("INSIDEAI_MAX_PROMPT_TOKENS", 64)
    max_new_tokens_cap: int = _env_int("INSIDEAI_MAX_NEW_TOKENS", 96)

    # Full per-head attention matrices are streamed while seq <= this cap;
    # beyond it only the head-mean matrix and per-head last rows are sent.
    full_attention_seq_cap: int = _env_int("INSIDEAI_FULL_ATTN_CAP", 64)

    # The MLP intermediate vector (3072 dims for distilgpt2) is mean-pooled
    # into this many contiguous channel groups before streaming.
    neuron_channels: int = _env_int("INSIDEAI_NEURON_CHANNELS", 128)

    # Diagram resolution for the layered MLP view: input/output columns and
    # the hidden column, plus the pooled |W| wiring between them.
    mlp_in_nodes: int = _env_int("INSIDEAI_MLP_IN_NODES", 12)
    mlp_hidden_nodes: int = _env_int("INSIDEAI_MLP_HIDDEN_NODES", 16)

    # How many dimensions of the positional-encoding vectors to stream raw.
    positional_dims: int = _env_int("INSIDEAI_POSITIONAL_DIMS", 24)

    topk_probs: int = _env_int("INSIDEAI_TOPK", 10)

    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            os.getenv(
                "INSIDEAI_CORS_ORIGINS",
                "http://localhost:3000,http://127.0.0.1:3000",
            ).split(",")
        )
    )

    # ------------------------------------------------------------- replay mode
    # Opt-in run recording (default OFF so normal use is byte-for-byte
    # unchanged). When on, every WS event of a generation is written to
    # results/runs/<date>_<sha>_<env>/ alongside a reproducibility manifest.
    record: bool = _env_flag("INSIDEAI_RECORD")

    # Start the server without loading a model — replay-only mode. Recorded
    # runs still list and replay over the same WS; live generation is refused.
    skip_model_load: bool = _env_flag("INSIDEAI_SKIP_MODEL")

    # Short label baked into each recording's directory name. Defaults to the
    # model's short name (e.g. "distilgpt2"); override for prod/dev/etc.
    env_tag: str = os.getenv("INSIDEAI_ENV", "").strip()

    # Where recordings live. Absolute so it's stable no matter the cwd.
    runs_dir: Path = field(default_factory=lambda: _REPO_ROOT / "results" / "runs")

    repo_root: Path = field(default_factory=lambda: _REPO_ROOT)


settings = Settings()
