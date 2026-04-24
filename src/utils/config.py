"""Minimal YAML config loader with dotted-merge and path interpolation."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: Dict[str, Any], over: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must be a mapping at the top level")
    return data


def _resolve_defaults(path: Path, seen: set[Path]) -> Dict[str, Any]:
    """Load `path` and recursively merge any `defaults:` it declares (left-to-right, later wins)."""
    path = path.resolve()
    if path in seen:
        raise RuntimeError(f"Circular defaults chain at {path}")
    seen.add(path)
    data = load_yaml(path)
    base: Dict[str, Any] = {}
    for ref in data.pop("defaults", []):
        ref_path = _resolve_ref(ref, path)
        base = _deep_merge(base, _resolve_defaults(ref_path, seen))
    return _deep_merge(base, data)


def load_config(
    experiment_cfg: str | os.PathLike,
    paths_cfg: str | os.PathLike | None = None,
    overrides: Iterable[str] = (),
) -> Dict[str, Any]:
    """Resolve one experiment YAML into a single flat dict.

    Resolution order (later wins):
      1. the transitively-resolved `defaults` chain of the experiment YAML
      2. the experiment YAML itself
      3. explicit `paths_cfg` if given
      4. CLI overrides as `dotted.key=value`
    """
    exp_path = Path(experiment_cfg).resolve()
    cfg = _resolve_defaults(exp_path, set())

    if paths_cfg is not None:
        cfg = _deep_merge(cfg, load_yaml(paths_cfg))

    for kv in overrides:
        if "=" not in kv:
            raise ValueError(f"Override must be dotted.key=value, got {kv!r}")
        key, raw = kv.split("=", 1)
        _set_dotted(cfg, key.strip(), _coerce(raw.strip()))

    cfg.setdefault("_meta", {})["experiment_config"] = str(exp_path)
    return cfg


def _resolve_ref(ref: str, anchor: Path) -> Path:
    p = Path(ref)
    if p.is_absolute() and p.exists():
        return p
    for cand in (anchor.parent / ref, REPO_ROOT / ref):
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError(f"Could not resolve config ref: {ref}")


def _set_dotted(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = cfg
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def _coerce(raw: str) -> Any:
    low = raw.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none"}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for p in dotted.split("."):
        if not isinstance(cur, Mapping) or p not in cur:
            return default
        cur = cur[p]
    return cur
