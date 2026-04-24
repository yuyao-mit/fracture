"""Run registry: append one row per run to metadata/runs/<benchmark_group>.csv.

Uses fcntl.flock for the read-modify-write so concurrent SLURM jobs finishing
against the same benchmark group don't corrupt the CSV.
"""
from __future__ import annotations

import csv
import fcntl
import json
import os
from pathlib import Path
from typing import Any, Mapping

try:
    from ..utils import REPO_ROOT
except ImportError:
    from utils import REPO_ROOT  # type: ignore


RUNS_DIR = REPO_ROOT / "metadata" / "runs"

FIELDS = [
    "run_name",
    "model_id",
    "seed",
    "split",
    "family",
    "config_path",
    "ckpt_path",
    "metrics_path",
    "status",
    "primary_score",
    "true_fem_used",
    "paper_eligible",
]


def _benchmark_group_for(split: str, family: str) -> str:
    """Map (family, split) -> benchmark_group file stem per EXPERIMENTS.md."""
    if split == "id":
        return "screen_id" if family == "baseline" else "main_id"
    if split.startswith("ood_"):
        return f"main_{split}"
    if split.startswith("lowdata"):
        return "main_lowdata" if family == "hybrid" else "screen_lowdata"
    if split.startswith("ablation_"):
        return split
    return split


def _csv_path(benchmark_group: str, runs_dir: Path) -> Path:
    return runs_dir / f"{benchmark_group}.csv"


def write_row(
    row: Mapping[str, Any],
    benchmark_group: str | None = None,
    runs_dir: Path | str = RUNS_DIR,
) -> Path:
    """Append/replace a row in metadata/runs/<benchmark_group>.csv keyed by run_name.

    Atomic under concurrent writes: holds an exclusive flock on a sibling lock
    file for the full read-modify-write.
    """
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    group = benchmark_group or _benchmark_group_for(row.get("split", ""), row.get("family", "baseline"))
    csv_path = _csv_path(group, runs_dir)
    lock_path = csv_path.with_suffix(csv_path.suffix + ".lock")

    run_name = row["run_name"]
    payload = {k: row.get(k, "") for k in FIELDS}

    with open(lock_path, "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            rows: list[dict[str, Any]] = []
            if csv_path.exists():
                with open(csv_path, newline="") as f:
                    rows = list(csv.DictReader(f))
            out = [{**r} for r in rows if r.get("run_name") != run_name]
            out.append(payload)
            tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
            with open(tmp_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(out)
            os.replace(tmp_path, csv_path)
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
    return csv_path


def dump_metrics(metrics_dir: Path | str, metrics: Mapping[str, Any]) -> Path:
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    p = metrics_dir / "metrics.json"
    with open(p, "w") as f:
        json.dump(dict(metrics), f, indent=2, default=float)
    return p
