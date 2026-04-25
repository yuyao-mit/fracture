import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from fem.adapter import FemConfig  # noqa: E402


def test_solver_config_pointer_is_followed(tmp_path):
    ref = tmp_path / "solver_my.yaml"
    ref.write_text(
        "material:\n  E: 1.0\n  nu: 0.3\n  Gc: 1.0\n  length_scale: 0.02\n"
        "solver:\n  nonlinear_max_iters: 50\n  nonlinear_tol: 1.0e-6\n"
        "mesh:\n  grid_nx: 128\n  grid_ny: 96\n"
    )
    cfg = {"solver": {"config": str(ref)}}
    fc = FemConfig.from_cfg(cfg)
    assert fc.E == 1.0
    assert fc.Gc == 1.0
    assert fc.ell == 0.02
    assert fc.max_stagger == 50
    assert fc.grid_nx == 128
    assert fc.grid_ny == 96


def test_inline_overrides_win_over_pointer(tmp_path):
    ref = tmp_path / "solver_my.yaml"
    ref.write_text("material:\n  E: 1.0\n")
    cfg = {"solver": {"config": str(ref), "material": {"E": 42.0}}}
    fc = FemConfig.from_cfg(cfg)
    assert fc.E == 42.0


def test_missing_pointer_file_raises(tmp_path):
    cfg = {"solver": {"config": str(tmp_path / "nope.yaml")}}
    with pytest.raises(FileNotFoundError):
        FemConfig.from_cfg(cfg)


def test_no_pointer_uses_defaults():
    fc = FemConfig.from_cfg({})
    assert fc.E == 210.0  # dataclass default


def test_flat_solver_keys_rejected():
    with pytest.raises(ValueError):
        FemConfig.from_cfg({"solver": {"nsteps": 10}})
