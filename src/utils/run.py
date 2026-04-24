"""Run naming and identity helpers aligned to EXPERIMENTS.md."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

_LOWER_RE = re.compile(r"^[a-z0-9_]+$")
_MIXED_RE = re.compile(r"^[A-Za-z0-9_]+$")  # model_id may be MixedCase (e.g. paramfem_targetGc)


@dataclass(frozen=True)
class RunIdentity:
    family: str          # baseline | hybrid
    task: str            # fracture
    split: str           # id | ood_geometry | ... | lowdataXX | ablation_<name>
    model_id: str        # fno | uno | codano | rno | paramfem | warmstart | <variant>
    seed: int            # 0 for current study

    @property
    def name(self) -> str:
        return f"{self.family}_{self.task}_{self.split}_{self.model_id}_s{self.seed}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.name,
            "family": self.family,
            "task": self.task,
            "split": self.split,
            "model_id": self.model_id,
            "seed": self.seed,
        }


_VALID_FAMILIES = {"baseline", "hybrid"}
_VALID_TASKS = {"fracture"}


def validate_part(kind: str, value: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"Invalid {kind}: must be a string, got {type(value).__name__}")
    pattern = _MIXED_RE if kind == "model_id" else _LOWER_RE
    if not pattern.match(value):
        raise ValueError(f"Invalid {kind} {value!r}: must match {pattern.pattern}")


def run_identity_from_cfg(cfg: Mapping[str, Any]) -> RunIdentity:
    run = cfg.get("run") or {}
    family = run.get("family") or cfg.get("family") or "baseline"
    task = run.get("task") or cfg.get("task") or "fracture"
    split = run.get("split") or cfg.get("split")
    model_id = run.get("model_id") or cfg.get("model_id") or (cfg.get("model") or {}).get("id")
    seed = run.get("seed") if run.get("seed") is not None else cfg.get("seed", 0)

    if split is None:
        raise ValueError("config missing 'run.split' (e.g. id, ood_geometry, lowdata10)")
    if model_id is None:
        raise ValueError("config missing 'run.model_id' (e.g. fno, paramfem)")

    for kind, val in [("family", family), ("task", task), ("split", split), ("model_id", model_id)]:
        validate_part(kind, val)

    if family not in _VALID_FAMILIES:
        raise ValueError(f"family must be one of {_VALID_FAMILIES}, got {family}")
    if task not in _VALID_TASKS:
        raise ValueError(f"task must be one of {_VALID_TASKS}, got {task}")

    return RunIdentity(family=family, task=task, split=split, model_id=model_id, seed=int(seed))
