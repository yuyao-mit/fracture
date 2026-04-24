"""Parameter-to-FEM hybrid: NN predicts a solver-compatible field, FEM solves.

A hybrid run is paper-eligible only when the FEM stack actually ran.
By default we fail fast if FEM is unavailable — set
`hybrid.allow_fallback=true` explicitly to run in neural-only mode
(useful for plumbing smoke tests; never for paper numbers).

Every step returns `true_fem_used`, which the trainer/evaluator propagate
to the registry and wandb so downstream analysis can filter to
paper-eligible runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn  # noqa: F401  (Tensor used via annotations)

try:
    from ..models import build_model
    from ..fem.adapter import (
        FemConfig,
        FemUnavailableError,
        is_fem_available,
        is_fem_solve_implemented,
        solve as fem_solve,
    )
    from .config import HybridConfig, HYBRID_TARGETS
except ImportError:
    from models import build_model  # type: ignore
    from fem.adapter import (  # type: ignore
        FemConfig,
        FemUnavailableError,
        is_fem_available,
        is_fem_solve_implemented,
        solve as fem_solve,
    )
    from hybrid.config import HybridConfig, HYBRID_TARGETS  # type: ignore


class HybridPredictor(nn.Module):
    """NN -> FEM pipeline. Fails fast if FEM unavailable unless allow_fallback."""

    def __init__(
        self,
        hybrid: HybridConfig,
        input_shape,
        output_shape,
        fem_cfg: FemConfig | None = None,
    ):
        super().__init__()
        self.hybrid = hybrid
        self.fem_cfg = fem_cfg or FemConfig()
        self.backbone = build_model(
            input_shape=input_shape,
            output_shape=output_shape,
            model_name=hybrid.backbone_id,
        )
        # A hybrid run is "FEM-ready" only if the stack is importable AND solve()
        # is actually implemented. `is_fem_available()` alone doesn't guarantee
        # solve() works (Phase-3 placeholder can be present with dolfinx installed).
        self._fem_ready = is_fem_available() and is_fem_solve_implemented()
        if not self._fem_ready and not hybrid.allow_fallback:
            raise FemUnavailableError(
                "Hybrid run requires a working FEM stack, but it is not ready: "
                "either dolfinx is missing or solve() is still a Phase-3 placeholder. "
                "Refusing to proceed silently. Either wire the FEM adapter and set "
                "is_fem_solve_implemented()=True, or set `hybrid.allow_fallback=true` "
                "explicitly — fallback runs are marked paper_eligible=false."
            )
        # Tracks whether FEM *actually ran* at least once this run. Only flipped
        # to True by a successful fem_solve() call inside step(). `fem_ready` is
        # NOT a proxy for this — see review comment on fem_ready vs true_fem_used.
        self.last_true_fem_used: bool = False
        self.any_true_fem_used: bool = False

    @property
    def fem_ready(self) -> bool:
        return self._fem_ready

    def forward(self, x: Tensor) -> Tensor:
        """Return the network output. FEM is applied in `step` (non-diff)."""
        return self.backbone(x)

    def step(self, x: Tensor) -> dict[str, Any]:
        latent = self.backbone(x)
        d: Tensor | None = None
        used = False
        if self._fem_ready:
            try:
                fields = {self.hybrid.target_field: latent.detach().cpu().numpy()}
                out = fem_solve(fields, self.fem_cfg, max_iters=self.hybrid.solver_steps)
                if "d" in out:
                    d = torch.from_numpy(out["d"]).to(latent.device, dtype=latent.dtype)
                used = True
            except FemUnavailableError:
                if not self.hybrid.allow_fallback:
                    raise
                used = False
        else:
            if not self.hybrid.allow_fallback:
                raise FemUnavailableError("FEM became unavailable mid-run")
        self.last_true_fem_used = used
        if used:
            self.any_true_fem_used = True
        return {"latent_field": latent, "d": d, "true_fem_used": used}


def build_hybrid_from_cfg(cfg, input_shape, output_shape) -> HybridPredictor:
    return HybridPredictor(
        hybrid=HybridConfig.from_cfg(cfg),
        input_shape=input_shape,
        output_shape=output_shape,
        fem_cfg=FemConfig.from_cfg(cfg),
    )
