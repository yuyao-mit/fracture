import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

spec = importlib.util.spec_from_file_location("submit_batch", REPO / "scripts" / "submit_batch.py")
sb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sb)


def test_overrides_without_equals_rejected():
    with pytest.raises(ValueError):
        sb._validate_overrides(["key_no_value"])


def test_overrides_with_comma_rejected():
    with pytest.raises(ValueError):
        sb._validate_overrides(["training.schedule=cos,warmup"])


def test_overrides_with_space_rejected():
    with pytest.raises(ValueError):
        sb._validate_overrides(["training.note=hello world"])


def test_clean_overrides_accepted():
    sb._validate_overrides(["training.epochs=5", "wandb.mode=offline"])


def test_stage_main_lowdata_includes_hybrid():
    cfgs = sb._configs_for_stage("main_lowdata", models=["fno"], fractions=["10", "25"], oods=None)
    names = [c.name for c in cfgs]
    assert "fno_10.yaml" in names
    assert "paramfem_10.yaml" in names
    assert "paramfem_25.yaml" in names


def test_stage_ood_includes_all_operators_and_hybrid():
    cfgs = sb._configs_for_stage("ood", models=["fno", "uno"], fractions=None, oods=["geometry"])
    names = [str(c) for c in cfgs]
    # 2 operators + 1 hybrid
    assert any("ood/geometry/fno.yaml" in n for n in names)
    assert any("ood/geometry/uno.yaml" in n for n in names)
    assert any("ood/geometry/paramfem.yaml" in n for n in names)
