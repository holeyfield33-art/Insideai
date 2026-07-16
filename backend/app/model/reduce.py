"""Tensor -> JSON-safe reductions.

Everything streamed to the frontend passes through this module. The rule is:
values are *reduced* (projected, pooled, quantized) but never invented — each
helper documents exactly what it preserves so the UI can label it honestly.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def round_floats(t: torch.Tensor, ndigits: int = 4) -> list:
    """Tensor -> nested lists with floats rounded for compact JSON."""
    factor = 10.0**ndigits
    return torch.round(t.detach().float() * factor).div(factor).tolist()


def quant255(t: torch.Tensor) -> list:
    """Quantize values already in [0, 1] (attention weights) to 0..255 ints.

    ~4x smaller on the wire than floats; 1/255 resolution is far below what a
    color ramp can show, so the visualization is unaffected. The frontend
    divides by 255 to recover the weight.
    """
    return (t.detach().float().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).tolist()


def pca_3d(x: torch.Tensor) -> list[list[float]]:
    """Project (n, d) vectors onto their top-3 principal components.

    Used to place token embeddings in the 3D embedding space. The projection
    is recomputed per step from the *actual* embedding matrix of the current
    sequence; the cloud is rescaled so its largest coordinate is 1 (the scene
    applies world scale). With n == 1 there is no variance — the point sits at
    the origin; with n == 2 the third component is zero-padded.
    """
    n = x.shape[0]
    if n == 1:
        return [[0.0, 0.0, 0.0]]
    xc = x.detach().float()
    xc = xc - xc.mean(dim=0, keepdim=True)
    # Full (thin) SVD: sequences are tiny (n <= ~112), so this is microseconds.
    u, s, _vh = torch.linalg.svd(xc, full_matrices=False)
    q = min(3, s.shape[0])
    pts = u[:, :q] * s[:q]
    if q < 3:
        pts = F.pad(pts, (0, 3 - q))
    peak = pts.abs().max()
    if peak > 1e-8:
        pts = pts / peak
    return [[round(v, 4) for v in row] for row in pts.tolist()]


def pool_channels(v: torch.Tensor, out_channels: int) -> torch.Tensor:
    """Mean-pool a (d,) activation vector into `out_channels` contiguous groups.

    3072 GELU activations -> 128 channels means each streamed value is the
    mean of 24 real neurons. Pooling preserves which *regions* of the MLP are
    active, which is what the neuron view renders.
    """
    d = v.shape[0]
    if d <= out_channels:
        return v.detach().float()
    return F.adaptive_avg_pool1d(v.detach().float().view(1, 1, d), out_channels).view(-1)


def block_mean_2d(m: torch.Tensor, rows: int, cols: int) -> list[list[float]]:
    """Pool a (R, C) matrix into (rows, cols) block means, normalized to 0..1.

    Used for MLP weight matrices: each cell is the mean |weight| of a real
    block of connections, so the diagram's edge strengths are the model's own
    wiring at reduced resolution.
    """
    pooled = F.adaptive_avg_pool2d(m.detach().float().unsqueeze(0).unsqueeze(0), (rows, cols))
    pooled = pooled.view(rows, cols)
    peak = pooled.max()
    if peak > 1e-12:
        pooled = pooled / peak
    return [[round(v, 3) for v in row] for row in pooled.tolist()]


def entropy_bits(probs: torch.Tensor) -> float:
    """Shannon entropy of a probability distribution, in bits."""
    p = probs.detach().float().clamp_min(1e-12)
    return float(-(p * torch.log2(p)).sum().item())


def row_entropy_bits(rows: torch.Tensor) -> list[float]:
    """Entropy in bits of each row of a (n, m) stochastic matrix."""
    p = rows.detach().float().clamp_min(1e-12)
    ent = -(p * torch.log2(p)).sum(dim=-1)
    return [round(v, 3) for v in ent.tolist()]


def norms(x: torch.Tensor, ndigits: int = 3) -> list[float]:
    """L2 norm of each row of (n, d)."""
    return [round(v, ndigits) for v in x.detach().float().norm(dim=-1).tolist()]


def fmt_bytes(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.0f}{unit}"
        num /= 1024
    return f"{num:.1f}TB"


def fmt_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    return f"{n / 1e3:.0f}K"
