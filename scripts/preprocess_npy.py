#!/usr/bin/env python
"""One-time preprocessing: convert compressed-float64 .npz trajectories into
uncompressed float32 per-array .npy "bundles" next to the originals.

Why: the raw files are DEFLATE-compressed float64; `np.load(npz)[key][slice]`
decompresses the *entire* ~1 GB `input_v2` array to read a 4-step window, so a
single sample costs ~11 s even warm. Writing each array as an uncompressed
float32 `.npy` lets the dataset `mmap` it and read only the bytes it needs
(~16 MB), cutting per-sample cost to tens of ms.

Layout (same dir as the .npz so split/lowdata stem logic is unchanged):
    train/<stem>.npz                      (kept; still the manifest source)
    train/<stem>.input_v2.npy   float32   (uncompressed)
    train/<stem>.target.npy     float32
    train/<stem>.ell.npy        float32
    train/<stem>.READY                    (sentinel; not globbed by the loader)

Full files only (matches include_partial=false). Resumable (skips READY).
"""
from __future__ import annotations

import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np

INPUT_KEYS = ("input_v2", "input", "inputs")
TARGET_KEYS = ("target", "targets")
ELL_KEYS = ("ell", "lc")
EXTRA_KEYS = ("latent_variables",)


def convert_one(args) -> tuple[str, str]:
    path, force = args
    base = path[:-len(".npz")]
    sentinel = base + ".READY"
    if os.path.exists(sentinel) and not force:
        return (os.path.basename(path), "skip")
    try:
        d = np.load(path)  # NpzFile; each d[key] decompresses that member once
        keys = set(d.keys())
        saved = []

        def _save(name):
            arr = np.asarray(d[name]).astype(np.float32)
            # tmp must end in .npy or np.save appends it (then os.replace can't find tmp)
            tmp = f"{base}.{name}.tmp.npy"
            np.save(tmp, arr)
            os.replace(tmp, f"{base}.{name}.npy")  # atomic; safe under --force reruns
            saved.append(name)

        ik = next((k for k in INPUT_KEYS if k in keys), None)
        if ik is None:
            return (os.path.basename(path), f"NO-INPUT-KEY keys={sorted(keys)}")
        _save(ik)
        tk = next((k for k in TARGET_KEYS if k in keys), None)
        if tk:
            _save(tk)
        ek = next((k for k in ELL_KEYS if k in keys), None)
        if ek:
            _save(ek)
        for k in EXTRA_KEYS:
            if k in keys:
                _save(k)

        with open(sentinel, "w") as f:
            f.write(",".join(saved))
        return (os.path.basename(path), "ok:" + ",".join(saved))
    except Exception as e:  # noqa: BLE001
        return (os.path.basename(path), f"ERR {type(e).__name__}: {str(e)[:120]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/pscratch/sd/y/yu_yao/MyQuota/fracture")
    ap.add_argument("--roles", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    tasks = []
    for role in args.roles:
        d = os.path.join(args.data_root, role)
        if not os.path.isdir(d):
            print(f"[skip] no dir {d}")
            continue
        files = sorted(p for p in glob.glob(os.path.join(d, "*.npz"))
                       if not p.endswith(".partial.npz"))
        print(f"[{role}] {len(files)} full .npz")
        tasks += [(p, args.force) for p in files]

    print(f"converting {len(tasks)} files with {args.workers} workers ...", flush=True)
    n_ok = n_skip = n_err = 0
    with Pool(args.workers) as pool:
        for i, (name, status) in enumerate(pool.imap_unordered(convert_one, tasks), 1):
            if status.startswith("ok"):
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_err += 1
                print(f"  [{i}/{len(tasks)}] {name}: {status}", flush=True)
            if i % 20 == 0 or i == len(tasks):
                print(f"  ...{i}/{len(tasks)}  ok={n_ok} skip={n_skip} err={n_err}", flush=True)
    print(f"DONE ok={n_ok} skip={n_skip} err={n_err}")


if __name__ == "__main__":
    main()
