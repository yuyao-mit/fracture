import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from utils.config import load_config, load_yaml, _coerce, _set_dotted  # noqa: E402


def test_coerce_primitives():
    assert _coerce("true") is True
    assert _coerce("False") is False
    assert _coerce("null") is None
    assert _coerce("42") == 42
    assert _coerce("3.14") == 3.14
    assert _coerce("fno") == "fno"


def test_set_dotted_creates_nested():
    d = {}
    _set_dotted(d, "a.b.c", 1)
    assert d == {"a": {"b": {"c": 1}}}


def test_load_id_fno_merges_defaults(tmp_path):
    cfg = load_config(str(REPO / "configs" / "experiments" / "id" / "fno.yaml"))
    assert cfg["run"]["family"] == "baseline"
    assert cfg["run"]["split"] == "id"
    assert cfg["run"]["seed"] == 0
    assert cfg["model"]["id"] == "fno"
    assert cfg["data"]["input_steps"] == 4
    assert cfg["training"]["epochs"] == 200


def test_overrides_apply():
    cfg = load_config(
        str(REPO / "configs" / "experiments" / "id" / "fno.yaml"),
        overrides=["training.epochs=5", "wandb.mode=offline"],
    )
    assert cfg["training"]["epochs"] == 5
    assert cfg["wandb"]["mode"] == "offline"


def test_hybrid_paramfem_id_config():
    cfg = load_config(str(REPO / "configs" / "experiments" / "id" / "paramfem.yaml"))
    assert cfg["run"]["family"] == "hybrid"
    assert cfg["run"]["model_id"] == "paramfem"
    assert cfg["hybrid"]["target_field"] == "Gc"
