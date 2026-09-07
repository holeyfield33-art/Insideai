"""Run recording — the flight recorder's black box.

When ``INSIDEAI_RECORD=1`` the backend captures every WebSocket event of a
generation, in emission order, to ``results/runs/<date>_<sha>_<env>/run.jsonl``
(one JSON event per line) plus a ``manifest.json`` reproducibility record.

Recording is passive: it observes the exact event dicts already being sent to
the client and never mutates them, so a recorded run and the live run that
produced it are identical on the wire. With recording OFF (the default) none of
this code runs during a generation, so live behavior is unchanged.

The first line of ``run.jsonl`` is the ``model_info`` event (normally sent once
on WS connect). Keeping it in the log makes a recording fully self-describing:
replay can re-emit it so the frontend sizes its per-layer arrays from the
recorded model, with no live model loaded.
"""

from __future__ import annotations

import json
import hashlib
import logging
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import settings

logger = logging.getLogger("insideai.recording")



def _slug(text: str) -> str:
    """Filesystem-safe short slug (model names carry '/', ':' etc.)."""
    text = text.strip().split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "run"


def _git_sha(cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        pass
    return None


def _pkg_commit(dist_name: str) -> str | None:
    """Read the git commit a distribution was installed from (PEP 610).

    pip writes ``direct_url.json`` with ``vcs_info.commit_id`` for git installs;
    this is the authoritative record of what's actually running. Unknown
    installed provenance stays null; declared pins are not execution evidence.
    """
    try:
        import importlib.metadata as im

        raw = im.distribution(dist_name).read_text("direct_url.json")
        if raw:
            info = json.loads(raw)
            commit = info.get("vcs_info", {}).get("commit_id")
            if commit:
                return commit
    except Exception:
        pass
    return None


def _pkg_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def _node_version() -> str | None:
    try:
        out = subprocess.run(
            ["node", "--version"], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        pass
    return None


def _repo_versions() -> dict:
    return {
        "python": platform.python_version(),
        "torch": _pkg_version("torch"),
        "transformers": _pkg_version("transformers"),
        "node": _node_version(),
    }


def _repo_shas() -> dict:
    return {
        "insideai": _git_sha(settings.repo_root),
        "unitarity-lab": _pkg_commit("unitarity-labs"),
        "VAR": _pkg_commit("var"),
    }


def _unique_dir(base: Path, name: str) -> Path:
    candidate = base / name
    suffix = 2
    while candidate.exists():
        candidate = base / f"{name}_{suffix}"
        suffix += 1
    return candidate


class RunRecorder:
    """Captures one generation to disk. One recorder per generation."""

    def __init__(self, run_dir: Path, model_info: dict, meta: dict):
        self.run_dir = run_dir
        self.model_info = model_info
        self._meta = meta
        self._started = time.time()
        self._steps = 0
        self._path = run_dir / "run.jsonl"
        self._fh = self._path.open("w", encoding="utf-8")
        # model_info is the first recorded event so the log is self-describing.
        self.record({"type": "model_info", **model_info})

    @classmethod
    def start(
        cls,
        *,
        model_info: dict,
        prompt: str,
        params,
        effective_seed: int | None,
        chat_mode: str,
        speed: float,
    ) -> "RunRecorder | None":
        """Create the run directory and open the log. Returns None (and logs)
        on any failure — recording must never break a live generation."""
        try:
            base = settings.runs_dir
            base.mkdir(parents=True, exist_ok=True)
            date = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            sha = (_git_sha(settings.repo_root) or "nogit")[:7]
            env = settings.env_tag or _slug(model_info.get("model", "model"))
            run_dir = _unique_dir(base, f"{date}_{sha}_{_slug(env)}")
            run_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "prompt": prompt,
                "model": model_info.get("model"),
                "chat_mode": chat_mode,
                "speed": speed,
                "params": {
                    "max_new_tokens": params.max_new_tokens,
                    "temperature": params.temperature,
                    "top_k": params.top_k,
                    "top_p": params.top_p,
                    "repetition_penalty": params.repetition_penalty,
                    "seed": params.seed,
                    "effective_seed": effective_seed,
                },
            }
            logger.info("recording run to %s", run_dir)
            return cls(run_dir, model_info, meta)
        except Exception:
            logger.exception("failed to start run recording")
            return None

    def record(self, event: dict) -> None:
        try:
            if event.get("type") == "step_end":
                self._steps += 1
            self._fh.write(json.dumps(event, separators=(",", ":")))
            self._fh.write("\n")
        except Exception:
            logger.exception("failed to record event %s", event.get("type"))

    def finalize(self, *, text: str, steps: int, reason: str) -> None:
        """Write manifest.json and close the log. Safe to call once."""
        try:
            self._fh.flush()
            with self._path.open("rb") as recorded:
                event_hash = hashlib.file_digest(recorded, "sha256").hexdigest()
            manifest = {
                "id": self.run_dir.name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt": self._meta["prompt"],
                "model": self._meta["model"],
                "chat_mode": self._meta["chat_mode"],
                "speed": self._meta["speed"],
                **self._meta["params"],
                "max_tokens": self._meta["params"]["max_new_tokens"],
                "steps": steps or self._steps,
                "reason": reason,
                "generated_text": text,
                "duration_s": round(time.time() - self._started, 2),
                "git_shas": _repo_shas(),
                "versions": _repo_versions(),
                "n_layer": self.model_info.get("n_layer"),
                "n_head": self.model_info.get("n_head"),
                "family": self.model_info.get("family"),
                "device": self.model_info.get("device"),
                "dtype": self.model_info.get("dtype"),
                "model_revision": self.model_info.get("model_revision"),
                "cuda_version": self.model_info.get("cuda_version"),
                "gpu_model": self.model_info.get("gpu_model"),
                "gpu_vram_bytes": self.model_info.get("gpu_vram_bytes"),
                "events_sha256": event_hash,
            }
            (self.run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
        except Exception:
            logger.exception("failed to write manifest")
        finally:
            try:
                self._fh.close()
            except Exception:
                pass


# ------------------------------------------------------------------- listing


def list_runs() -> list[dict]:
    """Every recorded run as a compact summary, newest first.

    Reads each run's manifest.json; runs without one (e.g. an interrupted
    recording still on disk) are skipped rather than surfaced half-formed.
    """
    base = settings.runs_dir
    if not base.exists():
        return []
    runs: list[dict] = []
    for run_dir in base.iterdir():
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file() or not (run_dir / "run.jsonl").is_file():
            continue
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("skipping unreadable manifest: %s", manifest_path)
            continue
        runs.append(
            {
                "id": run_dir.name,
                "prompt": m.get("prompt", ""),
                "model": m.get("model"),
                "timestamp": m.get("timestamp"),
                "steps": m.get("steps", 0),
            }
        )
    runs.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
    return runs


def load_run_events(run_id: str) -> list[dict]:
    """Recorded events for a run, in emission order (first line = model_info).

    ``run_id`` is validated to be a plain directory name — no traversal.
    Raises FileNotFoundError if the run doesn't exist.
    """
    if not run_id or "/" in run_id or "\\" in run_id or run_id in {".", ".."}:
        raise FileNotFoundError(run_id)
    run_dir = (settings.runs_dir / run_id).resolve()
    # Reject anything that escaped the runs directory.
    if settings.runs_dir.resolve() not in run_dir.parents:
        raise FileNotFoundError(run_id)
    log_path = run_dir / "run.jsonl"
    if not log_path.is_file():
        raise FileNotFoundError(run_id)
    events: list[dict] = []
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events
