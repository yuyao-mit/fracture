"""Config-driven training loop for baseline neural operators.

One Trainer = one run. The config block is the single source of truth for
what to train, on which split, where to write artifacts, and how to log.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from ..utils import load_paths, run_identity_from_cfg, wandb_from_cfg, ensure_dirs
    from ..data import resolve_split, build_datasets_for_split
    from ..models import build_from_cfg
    from ..evaluation import summarize, merge_prefix, write_row, dump_metrics
    from ..utils.loss import masked_mse
except ImportError:  # top-level-on-sys.path
    from utils import load_paths, run_identity_from_cfg, wandb_from_cfg, ensure_dirs  # type: ignore
    from data import resolve_split, build_datasets_for_split  # type: ignore
    from models import build_from_cfg  # type: ignore
    from evaluation import summarize, merge_prefix, write_row, dump_metrics  # type: ignore
    from utils.loss import masked_mse  # type: ignore


def _sample_shapes(sample: Mapping[str, torch.Tensor], batch_size: int, infer_latent: bool):
    T_in, C_in, H, W = sample["input_states"].shape
    if infer_latent:
        target_shape = sample["target"].shape
        T_out, C_out = 1, target_shape[0]
    else:
        target_shape = sample["target"].shape
        T_out, C_out = target_shape[0], target_shape[1]
    input_shape = (batch_size, T_in, C_in, H, W)
    output_shape = (batch_size, T_out, C_out, H, W)
    return input_shape, output_shape


def _mask_from_input(x: torch.Tensor, pred: torch.Tensor, infer_latent: bool, nan_ch: int = 9) -> torch.Tensor:
    nan_mask = x[:, -1, nan_ch]
    if infer_latent:
        return nan_mask.unsqueeze(1).expand_as(pred)
    return nan_mask.unsqueeze(1).unsqueeze(2).expand_as(pred)


def _load_ckpt(model, optimizer, ckpt_path: str | None, device):
    if not ckpt_path:
        return 0
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return int(ckpt.get("epoch", -1)) + 1


def train_from_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    ident = run_identity_from_cfg(cfg)
    paths = load_paths(cfg)

    tcfg = cfg.get("training") or {}
    dcfg = cfg.get("data") or {}
    epochs = int(tcfg.get("epochs", 200))
    batch_size = int(tcfg.get("batch_size", 8))
    lr = float(tcfg.get("lr", 1e-4))
    weight_decay = float(tcfg.get("weight_decay", 1e-5))
    num_workers = int(tcfg.get("num_workers", 4))
    ckpt_every = int(tcfg.get("ckpt_every", 10))
    device = tcfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    input_steps = int(dcfg.get("input_steps", 4))
    rollout_steps = int(dcfg.get("rollout_steps", 1))
    infer_latent = bool(dcfg.get("infer_latent_variable", True))
    nan_ch = int(dcfg.get("nan_channel", 9))

    split = resolve_split(ident.split, paths.data_root, seed=ident.seed)
    datasets = build_datasets_for_split(split, input_steps, rollout_steps, infer_latent)

    train_ds = datasets.get("train")
    val_ds = datasets.get("val")
    if train_ds is None:
        raise RuntimeError(f"no train split resolved from {paths.data_root}/train")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, persistent_workers=(num_workers > 0),
    )
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=True, persistent_workers=(num_workers > 0),
        )

    sample = train_ds[0]
    input_shape, output_shape = _sample_shapes(sample, batch_size, infer_latent)
    if ident.family == "hybrid":
        try:
            from ..hybrid import build_hybrid_from_cfg
        except ImportError:
            from hybrid import build_hybrid_from_cfg  # type: ignore
        model = build_hybrid_from_cfg(cfg, input_shape=input_shape, output_shape=output_shape).to(device)
    else:
        model = build_from_cfg(cfg, input_shape=input_shape, output_shape=output_shape).to(device)

    # `true_fem_used` is the *actual* FEM invocation flag, not a readiness flag.
    # - baseline runs: no FEM by definition -> False, but paper_eligible=True.
    # - hybrid runs:   flipped True only after `model.step()` calls `fem_solve()`
    #                  successfully. The current training loop calls `model(x)`
    #                  (forward) only, so hybrid runs necessarily end with
    #                  true_fem_used=False until Phase-3 wires step() into the loop.
    # This is intentional: paper_eligible for hybrid is the strict conjunction
    # (any FEM call actually succeeded), computed at the end of the run.

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    start_epoch = _load_ckpt(model, optimizer, tcfg.get("ckpt_path"), device)

    ckpt_dir = paths.ckpt_dir(ident.name)
    metric_dir = paths.metric_dir(ident.name)
    log_dir = paths.log_dir(ident.name)
    ensure_dirs(ckpt_dir, metric_dir, log_dir)

    wb = wandb_from_cfg(cfg, run_name=ident.name, log_dir=str(log_dir))
    wb.summary(ident.as_dict())
    wb.summary({
        "model_param_count": sum(p.numel() for p in model.parameters()),
    })

    best_val = float("inf")
    best_ckpt: Path | None = None

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"[Train {ident.name}] Epoch {epoch}"):
            x = batch["input_states"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            pred = model(x)
            if infer_latent:
                pred = pred.squeeze(1)
            mask = _mask_from_input(x, pred, infer_latent, nan_ch=nan_ch)
            loss = masked_mse(pred, y, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        val_metrics: dict[str, float] = {}
        if val_loader is not None:
            model.eval()
            accum = {"mse": 0.0, "mae": 0.0, "rel_l2": 0.0}
            n = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["input_states"].to(device, non_blocking=True)
                    y = batch["target"].to(device, non_blocking=True)
                    pred = model(x)
                    if infer_latent:
                        pred = pred.squeeze(1)
                    mask = _mask_from_input(x, pred, infer_latent, nan_ch=nan_ch)
                    m = summarize(pred, y, mask == 0)  # valid where mask==0 by convention
                    for k, v in m.items():
                        accum[k] = accum.get(k, 0.0) + float(v)
                    n += 1
            val_metrics = {k: v / max(1, n) for k, v in accum.items()}

        t1 = time.time()
        log = {
            "epoch": epoch,
            "train/mse": train_loss,
            "lr": lr,
            "runtime/epoch_s": t1 - t0,
        }
        log.update(merge_prefix("val", val_metrics))
        wb.log(log, step=epoch)

        print(f"[{ident.name}] epoch {epoch}/{epochs}  train_mse={train_loss:.6f}  "
              f"val_mse={val_metrics.get('mse', float('nan')):.6f}  "
              f"t={t1 - t0:.1f}s", flush=True)

        cur_val = val_metrics.get("mse", train_loss)
        is_best = cur_val < best_val
        if is_best:
            best_val = cur_val
            best_ckpt = ckpt_dir / "best.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_mse": cur_val,
                        "run": ident.as_dict()}, best_ckpt)

        if (epoch + 1) % ckpt_every == 0 or epoch == epochs - 1:
            save_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "run": ident.as_dict()}, save_path)

    # Compute FEM-usage flags at end of run, from *actual* invocations.
    if ident.family == "hybrid":
        true_fem_used = bool(getattr(model, "any_true_fem_used", False))
    else:
        true_fem_used = False
    paper_eligible = (ident.family == "baseline") or true_fem_used

    final_metrics = {"final/best_val_mse": best_val, **merge_prefix("final/val", val_metrics)}
    dump_metrics(metric_dir, {
        **ident.as_dict(),
        **final_metrics,
        "ckpt_path": str(best_ckpt) if best_ckpt else "",
        "true_fem_used": true_fem_used,
        "paper_eligible": paper_eligible,
    })
    wb.summary(final_metrics)
    wb.summary({"true_fem_used": true_fem_used, "paper_eligible": paper_eligible})

    write_row({
        "run_name": ident.name,
        "model_id": ident.model_id,
        "seed": ident.seed,
        "split": ident.split,
        "family": ident.family,
        "config_path": (cfg.get("_meta") or {}).get("experiment_config", ""),
        "ckpt_path": str(best_ckpt) if best_ckpt else "",
        "metrics_path": str(metric_dir / "metrics.json"),
        "status": "trained",
        "primary_score": f"{best_val:.6e}",
        "true_fem_used": str(true_fem_used).lower(),
        "paper_eligible": str(paper_eligible).lower(),
    })

    wb.finish()
    return {"best_val_mse": best_val, "run_name": ident.name,
            "ckpt_path": str(best_ckpt) if best_ckpt else ""}
