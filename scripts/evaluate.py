#!/usr/bin/env python
"""Thin entrypoint: config-driven test-set evaluation."""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from utils import load_config  # noqa: E402
from evaluation import evaluate_from_cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--paths", default=None, type=str)
    ap.add_argument("--ckpt", default=None, type=str,
                    help="Override checkpoint path. Defaults to ckpt_root/<run_name>/best.pt.")
    ap.add_argument("--save-predictions", action="store_true")
    ap.add_argument("overrides", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, paths_cfg=args.paths, overrides=args.overrides)
    out = evaluate_from_cfg(cfg, ckpt_path=args.ckpt, save_predictions=args.save_predictions)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
