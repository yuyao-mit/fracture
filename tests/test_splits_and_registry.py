import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from data.splits import resolve_split, _subsample  # noqa: E402
from evaluation.registry import write_row, _benchmark_group_for  # noqa: E402


def _mk_split_tree(root: Path, train_names, val_names, test_names):
    for sub, names in [("train", train_names), ("val", val_names), ("test", test_names)]:
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        for n in names:
            (d / f"{n}.npz").write_bytes(b"\x00")  # empty stand-in


def test_resolve_id_split(tmp_path):
    _mk_split_tree(tmp_path, ["a", "b"], ["v"], ["t"])
    s = resolve_split("id", tmp_path, seed=0, splits_dir=tmp_path / "_manifests")
    assert s.train_stems is None  # id = all
    assert s.train_dir == tmp_path / "train"


def test_resolve_lowdata_is_deterministic(tmp_path):
    names = [f"c{i:03d}" for i in range(20)]
    _mk_split_tree(tmp_path, names, ["v"], ["t"])
    a = resolve_split("lowdata25", tmp_path, seed=0, splits_dir=tmp_path / "_m")
    b = resolve_split("lowdata25", tmp_path, seed=0, splits_dir=tmp_path / "_m")
    assert a.train_stems == b.train_stems
    assert len(a.train_stems) == 5  # 25% of 20


def test_resolve_manifest_wins(tmp_path):
    _mk_split_tree(tmp_path, ["a", "b", "c"], ["v"], ["t1", "t2"])
    mdir = tmp_path / "_m"
    mdir.mkdir()
    (mdir / "ood_geometry.json").write_text(json.dumps({
        "train": ["a"],
        "val": ["v"],
        "test": ["t2"],
    }))
    s = resolve_split("ood_geometry", tmp_path, seed=0, splits_dir=mdir)
    assert s.train_stems == {"a"}
    assert s.test_stems == {"t2"}


def test_benchmark_group_mapping():
    assert _benchmark_group_for("id", "baseline") == "screen_id"
    assert _benchmark_group_for("id", "hybrid") == "main_id"
    assert _benchmark_group_for("ood_geometry", "baseline") == "main_ood_geometry"
    assert _benchmark_group_for("lowdata10", "baseline") == "screen_lowdata"
    assert _benchmark_group_for("lowdata10", "hybrid") == "main_lowdata"
    assert _benchmark_group_for("ablation_target", "hybrid") == "ablation_target"


def test_dataset_excludes_partial_by_default(tmp_path):
    # Build a folder with 1 full and 2 partial files; assert the dataset
    # ignores partials unless include_partial=True.
    sys_path_before = list(sys.path)
    try:
        import numpy as np
        from data.dataset import ChunkedScalarDatasetEfficient
    except Exception:
        import pytest as _pt
        _pt.skip("torch/numpy unavailable")
        return

    folder = tmp_path / "d"
    folder.mkdir()
    T, C, H, W = 6, 10, 8, 8
    # one full
    np.savez(folder / "shear_case_001_fields.npz",
             inputs=np.zeros((T, C, H, W), dtype=np.float32),
             targets=np.zeros((T, 1, H, W), dtype=np.float32),
             latent_variables=np.zeros((T, 1, H, W), dtype=np.float32),
             lc=np.zeros(T, dtype=np.float32))
    # two partials (same schema; filename suffix is the signal)
    for i in (2, 3):
        np.savez(folder / f"shear_case_00{i}_fields.partial.npz",
                 inputs=np.zeros((T, C, H, W), dtype=np.float32),
                 targets=np.zeros((T, 1, H, W), dtype=np.float32),
                 latent_variables=np.zeros((T, 1, H, W), dtype=np.float32),
                 lc=np.zeros(T, dtype=np.float32))

    ds_default = ChunkedScalarDatasetEfficient(folder=str(folder), input_steps=4, rollout_steps=1)
    assert len(ds_default.file_map) == 1

    ds_all = ChunkedScalarDatasetEfficient(folder=str(folder), input_steps=4, rollout_steps=1,
                                           include_partial=True)
    assert len(ds_all.file_map) == 3
    sys.path = sys_path_before


def test_write_row_updates_existing(tmp_path):
    row = {
        "run_name": "baseline_fracture_id_fno_s0",
        "model_id": "fno", "seed": 0, "split": "id", "family": "baseline",
        "config_path": "configs/experiments/id/fno.yaml",
        "ckpt_path": "/tmp/x.pt", "metrics_path": "/tmp/m.json",
        "status": "trained", "primary_score": "1.23e-3",
    }
    p = write_row(row, runs_dir=tmp_path)
    assert p.exists()
    # second write with same run_name should replace, not duplicate
    row2 = {**row, "primary_score": "4.56e-4"}
    write_row(row2, runs_dir=tmp_path)
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2   # header + one row
    assert "4.56e-4" in lines[1]
