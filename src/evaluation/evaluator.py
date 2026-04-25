"""Test-set evaluator for a trained baseline operator.

Reads a config + checkpoint, runs the test split, computes metrics,
writes metrics.json, appends to run registry, and logs to wandb (summary only).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from ..utils import load_paths, run_identity_from_cfg, wandb_from_cfg, ensure_dirs
    from ..data import resolve_split, build_datasets_for_split
    from ..models import build_from_cfg
    from ..evaluation import summarize, merge_prefix, write_row, dump_metrics
except ImportError:
    from utils import load_paths, run_identity_from_cfg, wandb_from_cfg, ensure_dirs  # type: ignore
    from data import resolve_split, build_datasets_for_split  # type: ignore
    from models import build_from_cfg  # type: ignore
    from evaluation import summarize, merge_prefix, write_row, dump_metrics  # type: ignore


def _mask_from_input(x: torch.Tensor, pred: torch.Tensor, infer_latent: bool, nan_ch: int = 9) -> torch.Tensor:
    nan_mask = x[:, -1, nan_ch]
    if infer_latent:
        return nan_mask.unsqueeze(1).expand_as(pred)
    return nan_mask.unsqueeze(1).unsqueeze(2).expand_as(pred)


def evaluate_from_cfg(
    cfg: Mapping[str, Any],
    ckpt_path: str | None = None,
    save_predictions: bool = False,
) -> dict[str, Any]:
    ident = run_identity_from_cfg(cfg)
    paths = load_paths(cfg)

    tcfg = cfg.get("training") or {}
    dcfg = cfg.get("data") or {}
    ecfg = cfg.get("eval") or {}
    batch_size = int(ecfg.get("batch_size", tcfg.get("batch_size", 8)))
    num_workers = int(ecfg.get("num_workers", tcfg.get("num_workers", 4)))
    device = ecfg.get("device", tcfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    input_steps = int(dcfg.get("input_steps", 4))
    rollout_steps = int(dcfg.get("rollout_steps", 1))
    infer_latent = bool(dcfg.get("infer_latent_variable", True))
    nan_ch = int(dcfg.get("nan_channel", 9))
    include_partial = bool(dcfg.get("include_partial", False))
    samples_cache_dir = dcfg.get("samples_cache_dir")

    split = resolve_split(ident.split, paths.data_root, seed=ident.seed)
    datasets = build_datasets_for_split(
        split, input_steps, rollout_steps, infer_latent,
        roles=("test",), include_partial=include_partial,
        samples_cache_dir=samples_cache_dir,
    )
    test_ds = datasets.get("test")
    if test_ds is None or len(test_ds) == 0:
        raise RuntimeError(f"no test split resolved for {ident.name}")

    loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=(num_workers > 0),
    )

    sample = test_ds[0]
    T_in, C_in, H, W = sample["input_states"].shape
    if infer_latent:
        C_out = sample["target"].shape[0]
        T_out = 1
    else:
        T_out, C_out = sample["target"].shape[:2]
    input_shape = (batch_size, T_in, C_in, H, W)
    output_shape = (batch_size, T_out, C_out, H, W)
    if ident.family == "hybrid":
        try:
            from ..hybrid import build_hybrid_from_cfg
        except ImportError:
            from hybrid import build_hybrid_from_cfg  # type: ignore
        model = build_hybrid_from_cfg(cfg, input_shape=input_shape, output_shape=output_shape).to(device)
    else:
        model = build_from_cfg(cfg, input_shape=input_shape, output_shape=output_shape).to(device)
    # Flags are computed after the eval loop from actual `model.step()` invocations.

    resolved_ckpt = ckpt_path or str(paths.ckpt_dir(ident.name) / "best.pt")
    ckpt = torch.load(resolved_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    metric_dir = paths.metric_dir(ident.name)
    pred_dir = paths.prediction_dir(ident.name)
    log_dir = paths.log_dir(ident.name)
    ensure_dirs(metric_dir, log_dir, pred_dir if save_predictions else metric_dir)

    wb = wandb_from_cfg(cfg, run_name=f"{ident.name}__eval", log_dir=str(log_dir))
    wb.summary(ident.as_dict())
    wb.summary({"eval/ckpt_path": resolved_ckpt})

    accum = {"mse": 0.0, "mae": 0.0, "rel_l2": 0.0}
    n = 0
    # `preds_out` stores the backbone's predicted parameter field (same space
    # as the dataset supervision target `latent_variables`, e.g. Gc). For
    # hybrid runs where FEM actually ran, `d_out` stores the FEM-solved damage
    # field per batch. To keep row alignment with `predictions.npy` we only
    # save `fem_damage.npy` when *every* batch produced a `d`. Partial fallback
    # (some batches missing d) skips the save with a warning.
    preds_out, targets_out, d_out = [], [], []
    n_batches_with_d = 0
    is_hybrid = (ident.family == "hybrid")
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[Eval {ident.name}]"):
            x = batch["input_states"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            # Hybrid: invoke step() so FEM (when ready) actually runs and the
            # any_true_fem_used flag is flipped. Baseline: plain forward.
            if is_hybrid:
                out = model.step(x)
                pred = out["latent_field"]
            else:
                pred = model(x)
                out = None
            if infer_latent:
                pred = pred.squeeze(1)
            mask = _mask_from_input(x, pred, infer_latent, nan_ch=nan_ch)
            # Scoring is against the supervision target `y`, which is the
            # parameter field the backbone predicts. Paper-eligible hybrid
            # runs should add damage metrics over `d` in a follow-up.
            m = summarize(pred, y, mask == 0)
            for k, v in m.items():
                accum[k] = accum.get(k, 0.0) + float(v)
            n += 1
            if save_predictions:
                preds_out.append(pred.cpu().numpy())
                targets_out.append(y.cpu().numpy())
                if is_hybrid and out is not None and out.get("d") is not None:
                    d_out.append(out["d"].cpu().numpy())
                    n_batches_with_d += 1

    test_metrics = {k: v / max(1, n) for k, v in accum.items()}

    # FEM-usage flags from *actual* invocations during this eval run.
    if ident.family == "hybrid":
        true_fem_used = bool(getattr(model, "any_true_fem_used", False))
    else:
        true_fem_used = False
    paper_eligible = (ident.family == "baseline") or true_fem_used

    wb.summary(merge_prefix("test", test_metrics))
    wb.summary({"true_fem_used": true_fem_used, "paper_eligible": paper_eligible})
    metrics_payload = {
        **ident.as_dict(),
        **merge_prefix("test", test_metrics),
        "eval/ckpt_path": resolved_ckpt,
        "true_fem_used": true_fem_used,
        "paper_eligible": paper_eligible,
    }
    dump_metrics(metric_dir, metrics_payload)

    if save_predictions:
        p = np.concatenate(preds_out, axis=0)
        t = np.concatenate(targets_out, axis=0)
        np.save(pred_dir / "predictions.npy", p)
        np.save(pred_dir / "targets.npy", t)
        if d_out:
            if n_batches_with_d == n:
                np.save(pred_dir / "fem_damage.npy", np.concatenate(d_out, axis=0))
            else:
                print(
                    f"[eval] skipped fem_damage.npy: only {n_batches_with_d}/{n} "
                    f"batches produced `d` (partial fallback). Row alignment with "
                    f"predictions.npy would be broken.",
                    flush=True,
                )

    write_row({
        "run_name": ident.name,
        "model_id": ident.model_id,
        "seed": ident.seed,
        "split": ident.split,
        "family": ident.family,
        "config_path": (cfg.get("_meta") or {}).get("experiment_config", ""),
        "ckpt_path": resolved_ckpt,
        "metrics_path": str(metric_dir / "metrics.json"),
        "status": "evaluated",
        "primary_score": f"{test_metrics.get('rel_l2', float('nan')):.6e}",
        "true_fem_used": str(true_fem_used).lower(),
        "paper_eligible": str(paper_eligible).lower(),
    })

    wb.finish()
    return {"run_name": ident.name, **merge_prefix("test", test_metrics)}
