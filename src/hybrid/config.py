"""Hybrid config dataclass. No torch import, safe to use in preflight/tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


HYBRID_TARGETS = {
    "paramfem": "Gc",
    "paramfem_targetGc": "Gc",
    "paramfem_targetdamage": "damage",
    "paramfem_latentfield": "latent",
}


@dataclass
class HybridConfig:
    backbone_id: str = "fno"
    target_field: str = "Gc"           # Gc | damage | latent
    parameterization: str = "field"     # field | lowrank | elementwise
    differentiable_fem: bool = False
    solver_steps: int = 1               # 1 | 3 | 10 | -1 (fullsolve)
    allow_fallback: bool = False        # must be explicit; fallback => not paper-eligible

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any]) -> "HybridConfig":
        hcfg = cfg.get("hybrid") or {}
        mcfg = cfg.get("model") or {}
        backbone = (mcfg.get("id") or hcfg.get("backbone") or "fno")
        return cls(
            backbone_id=backbone,
            target_field=hcfg.get("target_field", cls.target_field),
            parameterization=hcfg.get("parameterization", cls.parameterization),
            differentiable_fem=bool(hcfg.get("differentiable_fem", False)),
            solver_steps=int(hcfg.get("solver_steps", 1)),
            allow_fallback=bool(hcfg.get("allow_fallback", False)),
        )
