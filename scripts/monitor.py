#!/usr/bin/env python
"""One-shot monitor snapshot for submitted jobs.

Reads metadata/runs/submitted_jobs.csv, queries squeue for each job id,
prints a compact status table, tails latest stderr where available, and
reports whether the run's best checkpoint exists.

Usage:
    python scripts/monitor.py                 # snapshot once
    python scripts/monitor.py --stuck-tail 30 # tail 30 lines of stderr for non-RUNNING/completed
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
JOBS_CSV = _REPO / "metadata" / "runs" / "submitted_jobs.csv"
CKPT_ROOT = Path(os.environ.get("FRACTURE_CKPT_ROOT", "/ocean/projects/mch250029p/shared/ckpt"))


def _squeue_states() -> dict[str, str]:
    """Return {job_id: state} for this user's active jobs."""
    try:
        res = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "--noheader",
             "--format=%i|%T"],
            capture_output=True, text=True, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"[monitor] squeue unavailable: {e}", file=sys.stderr)
        return {}
    out = {}
    for line in res.stdout.splitlines():
        if "|" in line:
            jid, state = line.split("|", 1)
            out[jid.strip()] = state.strip()
    return out


def _sacct_final(job_id: str) -> str:
    try:
        res = subprocess.run(
            ["sacct", "-j", job_id, "--noheader", "--format=State", "--parsable2"],
            capture_output=True, text=True, check=False, timeout=10,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.splitlines()[0].strip()
    except Exception:
        pass
    return "UNKNOWN"


def _tail(path: Path, n: int) -> str:
    if not path.exists():
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 8192))
            tail = f.read().decode(errors="replace")
        return "\n".join(tail.splitlines()[-n:])
    except Exception as e:
        return f"<error reading {path}: {e}>"


def _ckpt_exists(run_name: str) -> bool:
    return (CKPT_ROOT / run_name / "best.pt").exists()


def _latest_stderr(log_dir: Path, job_id: str) -> Path | None:
    if not log_dir.exists():
        return None
    cand = log_dir / f"slurm-{job_id}.err"
    if cand.exists():
        return cand
    errs = sorted(log_dir.glob("slurm-*.err"))
    return errs[-1] if errs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(JOBS_CSV))
    ap.add_argument("--stuck-tail", type=int, default=0,
                    help="If >0, tail this many stderr lines for failed/unknown jobs")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[monitor] no jobs csv at {csv_path}")
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    active = _squeue_states()

    counts = {"PENDING": 0, "RUNNING": 0, "COMPLETED": 0, "FAILED": 0, "OTHER": 0, "UNKNOWN": 0}
    table = []
    failures = []
    for r in rows:
        jid = r.get("job_id", "").strip()
        run = r.get("run_name", "")
        log_dir = Path(r.get("log_dir", "") or "")
        if not jid or jid == "dry":
            state = "DRY"
        elif jid in active:
            state = active[jid]
        else:
            state = _sacct_final(jid)
        bucket = state if state in counts else "OTHER"
        counts[bucket] = counts.get(bucket, 0) + 1
        ckpt_ok = _ckpt_exists(run)
        table.append((jid, state, run, "ckpt:Y" if ckpt_ok else "ckpt:-", r.get("stage", "")))
        if state in {"FAILED", "TIMEOUT", "NODE_FAIL", "CANCELLED", "OUT_OF_MEMORY"}:
            failures.append((jid, run, log_dir))

    for jid, state, run, c, stage in table:
        print(f"  {jid:>10s}  {state:<12s}  {run:<55s}  {c}  [{stage}]")
    print("")
    print("summary: " + "  ".join(f"{k}={v}" for k, v in counts.items() if v))

    if args.stuck_tail and failures:
        print("\n--- tails of failed jobs ---")
        for jid, run, log_dir in failures:
            err = _latest_stderr(log_dir, jid)
            print(f"\n# {run}  (job {jid})")
            if err is None:
                print("  <no stderr found>")
            else:
                print(_tail(err, args.stuck_tail))


if __name__ == "__main__":
    main()
