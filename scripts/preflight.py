#!/usr/bin/env python
"""Preflight: fail fast before submitting a run.

Checks, in order:
  1. config resolves (defaults chain, run name)
  2. torch importable; cuda visibility (warning if missing, not error by default)
  3. dataset root + split role dirs readable; at least one .npz readable
  4. checkpoint and run-artifact dirs writable
  5. wandb mode is one of {null, online, offline, disabled}
  6. hybrid only: FEM availability matches hybrid.allow_fallback

Exit codes:
  0 = all checks passed
  2 = soft warning only (e.g. cuda missing on a login node)
  1 = hard failure
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


class PreflightError(RuntimeError):
    pass


def _ok(msg): print(f"[ OK ] {msg}")
def _warn(msg): print(f"[WARN] {msg}")
def _err(msg): print(f"[FAIL] {msg}")


def check_config(config_path: str, paths_override: str | None, overrides: list[str]):
    from utils import load_config, run_identity_from_cfg, load_paths
    cfg = load_config(config_path, paths_cfg=paths_override, overrides=overrides)
    ident = run_identity_from_cfg(cfg)
    paths = load_paths(cfg)
    _ok(f"config resolved: run_name={ident.name}")
    return cfg, ident, paths


def check_torch(strict_cuda: bool):
    try:
        import torch  # type: ignore
    except Exception as e:
        raise PreflightError(f"torch import failed: {e!r}")
    _ok(f"torch {torch.__version__}")
    if torch.cuda.is_available():
        _ok(f"cuda available (devices={torch.cuda.device_count()})")
    else:
        msg = "cuda NOT available (login node or missing driver)"
        if strict_cuda:
            raise PreflightError(msg)
        _warn(msg)
        return "cuda_missing"
    return None


def check_data(paths, ident):
    from data.splits import resolve_split
    data_root = Path(paths.data_root)
    if not data_root.exists():
        raise PreflightError(f"data_root does not exist: {data_root}")
    split = resolve_split(ident.split, data_root, seed=ident.seed)
    # At minimum train dir must be readable.
    train_dir = split.train_dir
    if not train_dir.is_dir():
        raise PreflightError(f"train dir not found: {train_dir}")
    files = [p for p in train_dir.iterdir() if p.suffix == ".npz"]
    if not files:
        raise PreflightError(f"no .npz in {train_dir}")
    # probe readability of one file
    f0 = files[0]
    try:
        with open(f0, "rb") as fh:
            fh.read(16)
    except PermissionError as e:
        raise PreflightError(
            f"data file unreadable: {f0} ({e}). Check ACLs: ls -la {f0}; "
            f"likely `chgrp -R mch250029p {data_root}` or add account to the owning group."
        )
    _ok(f"data readable ({len(files)} files in {train_dir.name}, sample ok: {f0.name})")


def check_io_dirs(paths, ident):
    for d in (paths.ckpt_dir(ident.name), paths.log_dir(ident.name), paths.metric_dir(ident.name)):
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".preflight"
            probe.write_text("")
            probe.unlink()
        except Exception as e:
            raise PreflightError(f"cannot write to {d}: {e}")
    _ok("ckpt/log/metric dirs writable")


def check_wandb(cfg):
    w = cfg.get("wandb") or {}
    mode = w.get("mode")
    if mode not in (None, "online", "offline", "disabled"):
        raise PreflightError(f"wandb.mode invalid: {mode!r}")
    enabled = bool(w.get("enabled", True))
    _ok(f"wandb enabled={enabled} mode={mode}")


def check_hybrid(cfg, ident):
    if ident.family != "hybrid":
        return
    # Mirror the HybridPredictor gate exactly: importable AND solve() is Phase-3-ready.
    from fem.adapter import is_fem_available, is_fem_solve_implemented
    from hybrid import HybridConfig
    hcfg = HybridConfig.from_cfg(cfg)
    dolfinx_ok = is_fem_available()
    solve_ok = is_fem_solve_implemented()
    fem_ok = dolfinx_ok and solve_ok
    if fem_ok:
        _ok("hybrid: FEM stack available (dolfinx + solve implemented)")
        return
    reason = (
        "dolfinx not importable" if not dolfinx_ok
        else "dolfinx importable but solve() is still a Phase-3 placeholder"
    )
    if hcfg.allow_fallback:
        _warn(f"hybrid: FEM not ready ({reason}); allow_fallback=true → run will NOT be paper-eligible")
        return
    raise PreflightError(
        f"hybrid run requires FEM but it is not ready: {reason}. "
        "Either wire the FEM adapter (is_fem_solve_implemented()==True) or set "
        "hybrid.allow_fallback=true to run as a non-paper-eligible smoke."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--paths", default=None)
    ap.add_argument("--strict-cuda", action="store_true")
    ap.add_argument("overrides", nargs="*", default=[])
    args = ap.parse_args()

    any_warn = False
    try:
        cfg, ident, paths = check_config(args.config, args.paths, args.overrides)
        warn = check_torch(strict_cuda=args.strict_cuda)
        if warn == "cuda_missing":
            any_warn = True
        check_data(paths, ident)
        check_io_dirs(paths, ident)
        check_wandb(cfg)
        check_hybrid(cfg, ident)
    except PreflightError as e:
        _err(str(e))
        sys.exit(1)
    sys.exit(2 if any_warn else 0)


if __name__ == "__main__":
    main()
