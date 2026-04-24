import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from utils.run import run_identity_from_cfg, validate_part  # noqa: E402
from utils.paths import load_paths  # noqa: E402


def test_run_name_format():
    cfg = {
        "run": {"family": "baseline", "task": "fracture", "split": "id",
                "model_id": "fno", "seed": 0},
    }
    ident = run_identity_from_cfg(cfg)
    assert ident.name == "baseline_fracture_id_fno_s0"


def test_run_name_lowdata_hybrid():
    cfg = {
        "run": {"family": "hybrid", "task": "fracture", "split": "lowdata10",
                "model_id": "paramfem", "seed": 0},
    }
    ident = run_identity_from_cfg(cfg)
    assert ident.name == "hybrid_fracture_lowdata10_paramfem_s0"


def test_run_name_ablation():
    cfg = {
        "run": {"family": "hybrid", "task": "fracture", "split": "ablation_target",
                "model_id": "paramfem_targetgc", "seed": 0},
    }
    ident = run_identity_from_cfg(cfg)
    assert ident.name == "hybrid_fracture_ablation_target_paramfem_targetgc_s0"


def test_invalid_parts_rejected():
    with pytest.raises(ValueError):
        validate_part("split", "Bad-Name")
    with pytest.raises(ValueError):
        validate_part("family", "Baseline")  # family must be lowercase


def test_model_id_allows_mixed_case():
    # EXPERIMENTS.md uses names like 'paramfem_targetGc' — must be accepted.
    validate_part("model_id", "paramfem_targetGc")
    validate_part("model_id", "fno")


def test_load_paths_has_expected_layout():
    paths = load_paths({})
    for attr in ("data_root", "ckpt_root", "run_root",
                 "log_root", "metric_root", "prediction_root", "figure_root", "table_root"):
        assert getattr(paths, attr) is not None
    assert str(paths.ckpt_dir("run_x")).endswith("run_x")
    assert str(paths.log_dir("run_y")).endswith("run_y")
