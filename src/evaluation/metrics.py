"""Field and fracture-specific metrics for fracture PDE prediction.

All functions accept torch tensors and an optional boolean `valid_mask`
(True where the point is a real domain point, False where it should be
ignored, e.g. NaN regions). Shapes follow the dataset convention
`[..., H, W]` or `[..., C, H, W]`.
"""
from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor


def _as_bool_mask(mask: Tensor | None, like: Tensor) -> Tensor:
    if mask is None:
        return torch.ones_like(like, dtype=torch.bool)
    m = mask.to(dtype=torch.bool)
    if m.shape != like.shape:
        m = m.expand_as(like)
    return m


def masked_mse(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    m = _as_bool_mask(mask, pred)
    diff = (pred - target) ** 2
    denom = m.sum().clamp(min=1)
    return (diff * m).sum() / denom


def masked_relative_l2(pred: Tensor, target: Tensor, mask: Tensor | None = None, eps: float = 1e-8) -> Tensor:
    m = _as_bool_mask(mask, pred)
    num = ((pred - target) ** 2 * m).sum()
    den = ((target ** 2) * m).sum().clamp(min=eps)
    return (num / den).sqrt()


def masked_mae(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    m = _as_bool_mask(mask, pred)
    denom = m.sum().clamp(min=1)
    return ((pred - target).abs() * m).sum() / denom


def damage_iou(pred_d: Tensor, target_d: Tensor, threshold: float = 0.5, mask: Tensor | None = None) -> Tensor:
    """IoU of the thresholded damage field (rough crack-region proxy)."""
    p = (pred_d > threshold)
    t = (target_d > threshold)
    if mask is not None:
        m = mask.to(dtype=torch.bool).expand_as(p)
        p = p & m
        t = t & m
    inter = (p & t).sum().float()
    union = (p | t).sum().float().clamp(min=1)
    return inter / union


def irreversibility_violation(pred_d_t: Tensor, pred_d_tm1: Tensor, mask: Tensor | None = None) -> Tensor:
    """Fraction of points where damage decreases across consecutive steps."""
    viol = (pred_d_t < pred_d_tm1 - 1e-6)
    if mask is not None:
        viol = viol & mask.to(dtype=torch.bool).expand_as(viol)
        denom = mask.sum().clamp(min=1)
    else:
        denom = torch.tensor(float(viol.numel()), device=viol.device)
    return viol.float().sum() / denom


def nonphysical_rate(pred: Tensor, lo: float = 0.0, hi: float = 1.0, mask: Tensor | None = None) -> Tensor:
    """Fraction of points outside the admissible [lo, hi] range."""
    out = (pred < lo) | (pred > hi)
    if mask is not None:
        out = out & mask.to(dtype=torch.bool).expand_as(out)
        denom = mask.sum().clamp(min=1)
    else:
        denom = torch.tensor(float(out.numel()), device=out.device)
    return out.float().sum() / denom


def summarize(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> dict[str, float]:
    out = {
        "mse": masked_mse(pred, target, mask).item(),
        "mae": masked_mae(pred, target, mask).item(),
        "rel_l2": masked_relative_l2(pred, target, mask).item(),
    }
    # Damage-like channel: bounded in [0,1]. Only add if the target seems to be damage.
    try:
        if float(target.min()) >= -1e-3 and float(target.max()) <= 1.0 + 1e-3:
            out["damage_iou"] = damage_iou(pred, target, mask=mask).item()
            out["nonphysical_rate"] = nonphysical_rate(pred, mask=mask).item()
    except RuntimeError:
        pass
    return out


def merge_prefix(prefix: str, d: Mapping[str, float]) -> dict[str, float]:
    return {f"{prefix}/{k}": v for k, v in d.items()}
