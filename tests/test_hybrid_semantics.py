import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from hybrid.config import HybridConfig  # no torch
from fem.adapter import FemUnavailableError, is_fem_available, is_fem_solve_implemented

try:
    import torch  # noqa: F401
    # Actually instantiate something to catch broken installs (missing .so etc.)
    torch.zeros(1)
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not importable in this env")


def test_hybrid_config_defaults_forbid_fallback():
    h = HybridConfig.from_cfg({})
    assert h.allow_fallback is False


def test_hybrid_config_reads_flag():
    h = HybridConfig.from_cfg({"hybrid": {"allow_fallback": True}})
    assert h.allow_fallback is True


def test_fem_solve_unimplemented_until_phase3():
    # Protects the `paper_eligible` invariant: the FEM-ready gate is FALSE
    # until solve() is actually implemented, regardless of dolfinx import.
    assert is_fem_solve_implemented() is False


@requires_torch
def test_hybrid_refuses_to_build_when_fem_missing_and_no_fallback():
    if not HAS_TORCH:
        pytest.skip("torch unavailable")
    if is_fem_available() and is_fem_solve_implemented():
        pytest.skip("FEM fully available in this env; fail-fast path is not exercisable")
    from hybrid.paramfem import HybridPredictor
    h = HybridConfig(backbone_id="fno", allow_fallback=False)
    with pytest.raises(FemUnavailableError):
        HybridPredictor(hybrid=h, input_shape=(1, 4, 10, 32, 32), output_shape=(1, 1, 1, 32, 32))


@requires_torch
def test_hybrid_allow_fallback_builds_without_fem():
    if not HAS_TORCH:
        pytest.skip("torch unavailable")
    if is_fem_available() and is_fem_solve_implemented():
        pytest.skip("FEM fully available; fallback gate is not exercisable")
    from hybrid.paramfem import HybridPredictor
    h = HybridConfig(backbone_id="fno", allow_fallback=True)
    m = HybridPredictor(hybrid=h, input_shape=(1, 4, 10, 32, 32), output_shape=(1, 1, 1, 32, 32))
    assert m.fem_ready is False
    assert m.last_true_fem_used is False
    assert m.any_true_fem_used is False
