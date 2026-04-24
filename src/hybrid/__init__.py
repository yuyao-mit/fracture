"""Hybrid neural-network and FEM coupling modules.

HybridConfig is torch-free. HybridPredictor + build_hybrid_from_cfg are lazy
because they pull torch + backbone models.
"""
from .config import HybridConfig, HYBRID_TARGETS

__all__ = [
    "HybridConfig",
    "HYBRID_TARGETS",
    "HybridPredictor",
    "build_hybrid_from_cfg",
]


def __getattr__(name):
    if name in {"HybridPredictor", "build_hybrid_from_cfg"}:
        from . import paramfem
        return getattr(paramfem, name)
    raise AttributeError(name)
