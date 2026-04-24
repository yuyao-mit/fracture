"""Resolve shared-storage paths for runs, checkpoints, and artifacts."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import REPO_ROOT, load_yaml


DEFAULT_PATHS_YAML = REPO_ROOT / "configs" / "paths" / "shared_paths.yaml"
EXAMPLE_PATHS_YAML = REPO_ROOT / "configs" / "paths" / "shared_paths.example.yaml"


@dataclass(frozen=True)
class Paths:
    data_root: Path
    ckpt_root: Path
    run_root: Path
    log_root: Path
    metric_root: Path
    prediction_root: Path
    figure_root: Path
    table_root: Path

    def ckpt_dir(self, run_name: str) -> Path:
        return self.ckpt_root / run_name

    def log_dir(self, run_name: str) -> Path:
        return self.log_root / run_name

    def metric_dir(self, run_name: str) -> Path:
        return self.metric_root / run_name

    def prediction_dir(self, run_name: str) -> Path:
        return self.prediction_root / run_name

    def figure_dir(self, run_name: str) -> Path:
        return self.figure_root / run_name

    def table_dir(self, benchmark_group: str) -> Path:
        return self.table_root / benchmark_group

    def split_dir(self, split: str) -> Path:
        # id -> train/val/test; ood_*/lowdata* handled by split manifests
        return self.data_root / split


def _env_override(key: str) -> str | None:
    return os.environ.get(key)


def load_paths(cfg: Mapping[str, Any] | None = None) -> Paths:
    """Resolve paths: cfg['paths'] > env vars > configs/paths/shared_paths.yaml > example."""
    p: dict[str, Any] = {}

    if DEFAULT_PATHS_YAML.exists():
        p.update(load_yaml(DEFAULT_PATHS_YAML).get("paths", {}))
    elif EXAMPLE_PATHS_YAML.exists():
        p.update(load_yaml(EXAMPLE_PATHS_YAML).get("paths", {}))

    for key, env in [
        ("data_root", "FRACTURE_DATA_ROOT"),
        ("ckpt_root", "FRACTURE_CKPT_ROOT"),
        ("run_root", "FRACTURE_RUN_ROOT"),
    ]:
        v = _env_override(env)
        if v:
            p[key] = v

    if cfg is not None and isinstance(cfg.get("paths"), Mapping):
        p.update(cfg["paths"])

    run_root = Path(p.get("run_root", "/ocean/projects/mch250029p/shared/experiments/fracture"))
    return Paths(
        data_root=Path(p.get("data_root", "/ocean/projects/mch250029p/shared/fracture")),
        ckpt_root=Path(p.get("ckpt_root", "/ocean/projects/mch250029p/shared/ckpt")),
        run_root=run_root,
        log_root=Path(p.get("log_root", run_root / "logs")),
        metric_root=Path(p.get("metric_root", run_root / "metrics")),
        prediction_root=Path(p.get("prediction_root", run_root / "predictions")),
        figure_root=Path(p.get("figure_root", run_root / "figures")),
        table_root=Path(p.get("table_root", run_root / "tables")),
    )


def ensure_dirs(*dirs: Path | str) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
