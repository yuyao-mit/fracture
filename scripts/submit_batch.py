#!/usr/bin/env python
"""Submit a batch of experiment configs to SLURM and log a mapping.

Example:
    # ID screening (all four operators)
    python scripts/submit_batch.py --stage screening_id

    # Low-data screening (10/25/100)
    python scripts/submit_batch.py --stage screening_lowdata

    # OOD for explicit model ids (e.g. best two from screening)
    python scripts/submit_batch.py --stage ood --models fno uno

    # Dry-run (print only)
    python scripts/submit_batch.py --stage screening_id --dry-run

Writes: metadata/runs/submitted_jobs.csv  (append-safe, keyed by run_name+job_id)
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_SRC = _REPO / "src"
sys.path.insert(0, str(_SRC))

from utils import load_config, run_identity_from_cfg, load_paths  # noqa: E402


OPS = ["fno", "uno", "codano", "rno"]
SUBMIT_SH = _HERE / "submit.sh"
JOBS_CSV = _REPO / "metadata" / "runs" / "submitted_jobs.csv"
JOBS_FIELDS = ["submitted_at", "stage", "run_name", "model_id", "split",
               "config_path", "job_id", "log_dir", "status"]


def _configs_for_stage(stage: str, models: list[str] | None, fractions: list[str] | None,
                       oods: list[str] | None) -> list[Path]:
    models = models or OPS
    fractions = fractions or ["10", "25", "100"]
    oods = oods or ["geometry", "load", "material", "resolution"]

    if stage == "screening_id":
        return [_REPO / f"configs/experiments/id/{m}.yaml" for m in models]
    if stage == "screening_lowdata":
        return [_REPO / f"configs/experiments/low_data/{m}_{f}.yaml"
                for m in models for f in fractions]
    if stage == "main":
        out = [_REPO / f"configs/experiments/id/{m}.yaml" for m in models]
        out.append(_REPO / "configs/experiments/id/paramfem.yaml")
        return out
    if stage == "main_lowdata":
        # Layer 2-C: selected baselines + paramfem across all five fractions.
        lowdata_fracs = fractions if fractions else ["05", "10", "25", "50", "100"]
        out = [_REPO / f"configs/experiments/low_data/{m}_{f}.yaml"
               for m in models for f in lowdata_fracs]
        out += [_REPO / f"configs/experiments/low_data/paramfem_{f}.yaml"
                for f in lowdata_fracs]
        return out
    if stage == "ood":
        out = [_REPO / f"configs/experiments/ood/{ood}/{m}.yaml"
               for ood in oods for m in models]
        # hybrid paramfem on each OOD split
        for ood in oods:
            out.append(_REPO / f"configs/experiments/ood/{ood}/paramfem.yaml")
        return out
    if stage == "ablations":
        ab_dir = _REPO / "configs/experiments/ablations"
        return sorted(p for p in ab_dir.glob("*.yaml") if "template" not in p.name)
    raise ValueError(f"unknown stage {stage!r}")


def _validate_overrides(overrides: list[str]) -> None:
    """`sbatch --export=ALL,K=V,K=V,...` splits on commas. Reject any token
    containing a comma or a shell metachar so the export string can't be
    silently misparsed into an extra env var or interpreted by the shell."""
    bad_chars = set(",$`\\\"' ")
    for o in overrides:
        if "=" not in o:
            raise ValueError(f"override must be dotted.key=value, got {o!r}")
        if any(c in bad_chars for c in o):
            raise ValueError(
                f"override {o!r} contains a character from {sorted(bad_chars)!r} "
                f"which would corrupt the sbatch --export list"
            )


def _submit_one(cfg_path: Path, overrides: list[str], account: str, env: str,
                dry_run: bool) -> dict:
    _validate_overrides(overrides)
    cfg = load_config(str(cfg_path), overrides=overrides)
    ident = run_identity_from_cfg(cfg)
    paths = load_paths(cfg)
    log_dir = paths.log_dir(ident.name)
    log_dir.mkdir(parents=True, exist_ok=True)

    exports = [
        f"CONFIG={cfg_path}",
        f"RUN_NAME={ident.name}",
        f"RUN_LOG_DIR={log_dir}",
        f"CONDA_ENV={env}",
    ]
    if overrides:
        exports.append("OVERRIDES=" + " ".join(overrides))

    cmd = [
        "sbatch",
        f"--job-name=frac-{ident.model_id}",
        f"--account={account}",
        f"--output={log_dir}/slurm-%j.out",
        f"--error={log_dir}/slurm-%j.err",
        f"--export=ALL,{','.join(exports)}",
        str(SUBMIT_SH),
    ]

    if dry_run:
        print("DRY: " + " ".join(cmd))
        return {"run_name": ident.name, "config_path": str(cfg_path),
                "model_id": ident.model_id, "split": ident.split,
                "log_dir": str(log_dir), "job_id": "dry", "status": "dry"}

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"sbatch FAILED for {ident.name}: {res.stderr.strip()}", file=sys.stderr)
        return {"run_name": ident.name, "config_path": str(cfg_path),
                "model_id": ident.model_id, "split": ident.split,
                "log_dir": str(log_dir), "job_id": "",
                "status": f"submit_error: {res.stderr.strip()[:200]}"}
    job_id = res.stdout.strip().split()[-1]
    print(f"[sbatch] {ident.name} -> job {job_id}")
    return {"run_name": ident.name, "config_path": str(cfg_path),
            "model_id": ident.model_id, "split": ident.split,
            "log_dir": str(log_dir), "job_id": job_id, "status": "submitted"}


def _append_rows(rows: list[dict], stage: str):
    JOBS_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = JOBS_CSV.exists()
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(JOBS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=JOBS_FIELDS)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({"submitted_at": now, "stage": stage, **r})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["screening_id", "screening_lowdata", "main", "main_lowdata",
                             "ood", "ablations"])
    ap.add_argument("--models", nargs="*", default=None,
                    help="Subset of {fno,uno,codano,rno}")
    ap.add_argument("--fractions", nargs="*", default=None,
                    help="Low-data fractions like 10 25 100")
    ap.add_argument("--oods", nargs="*", default=None,
                    help="OOD subdirs like geometry load material resolution")
    ap.add_argument("--account", default="mch250029p")
    ap.add_argument("--env", default="ai4phasefield")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("overrides", nargs="*", default=[])
    args = ap.parse_args()

    configs = _configs_for_stage(args.stage, args.models, args.fractions, args.oods)
    missing = [c for c in configs if not c.exists()]
    if missing:
        print("Missing config files:\n  " + "\n  ".join(str(m) for m in missing), file=sys.stderr)
        sys.exit(3)

    rows = []
    for c in configs:
        rows.append(_submit_one(c, args.overrides, args.account, args.env, args.dry_run))

    if not args.dry_run:
        _append_rows(rows, args.stage)
        print(f"[batch] recorded {len(rows)} rows -> {JOBS_CSV}")


if __name__ == "__main__":
    main()
