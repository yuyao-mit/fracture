
# Copyright (c) yu_yao@mit.edu
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.loss import rollout_mse_loss
from utils.dataset import FractureDataset
from models import build_model


############# Accuracy settings #############
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(description="Training on SINGLE GPU")
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--dataset_dir", required=True, type=str)
    parser.add_argument("--log_dir", required=True, type=str)
    parser.add_argument("--ckpt_dir", required=True, type=str)
    parser.add_argument("--log_every", default=1, type=int)
    parser.add_argument("--ckpt_every", default=10, type=int)
    parser.add_argument("--val_every", default=1, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--rollout_steps", default=3, type=int)
    parser.add_argument("--ckpt_path", default=None, type=str)

    # wandb options
    parser.add_argument("--wandb_project", default="fracture_pred", type=str)
    parser.add_argument("--wandb_entity", default=None, type=str)
    parser.add_argument("--wandb_run_name", default=None, type=str)
    parser.add_argument("--wandb_mode", default=None, type=str)  # "online" / "offline" / "disabled"

    return parser.parse_args()


def _load_ckpt(model, optimizer, ckpt_path, device):
    """
    Load checkpoint if provided. Returns start_epoch.
    """
    if ckpt_path is None:
        return 0

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"ckpt_path not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=True)
    else:
        raise KeyError("Checkpoint missing key: 'model_state'")

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    print(f"[Resume] Loaded ckpt from {ckpt_path}, start_epoch={start_epoch}")
    return start_epoch


def train(
    model,
    epochs,
    batch_size,
    dataset_dir,
    log_dir,
    ckpt_dir,
    log_every=10,
    ckpt_every=1,
    val_every=1,
    device="cuda",
    rollout_steps=3,
    num_workers=4,
    ckpt_path=None,
):

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_dataset = FractureDataset(os.path.join(dataset_dir, "train"), rollout_steps=rollout_steps)
    val_dataset = FractureDataset(os.path.join(dataset_dir, "val"), rollout_steps=rollout_steps)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[Warn] CUDA not available, switch to CPU.")

    model.to(device)
    start_epoch = _load_ckpt(model, optimizer, ckpt_path, device)

    # ---- Training loop ----
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss_epoch = 0.0

        pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch}")
        for it, batch in enumerate(pbar):
            x = batch["image"].to(device, non_blocking=True)  # [B,C,H,W]
            y = batch["mask"].to(device, non_blocking=True)   # [B,K,1,H,W]

            optimizer.zero_grad(set_to_none=True)
            loss, _ = rollout_mse_loss(model, x, y, detach=False)
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()

            if (it + 1) % log_every == 0:
                pbar.set_postfix(loss=loss.item())

        train_loss_epoch /= max(1, len(train_loader))

        # ---- Validation (epoch-level by default) ----
        model.eval()
        val_loss_epoch = 0.0
        if (epoch + 1) % val_every == 0:
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["image"].to(device, non_blocking=True)
                    y = batch["mask"].to(device, non_blocking=True)
                    loss, _ = rollout_mse_loss(model, x, y, detach=False)
                    val_loss_epoch += loss.item()

            val_loss_epoch /= max(1, len(val_loader))
        else:
            val_loss_epoch = float("nan")

        # ---- Console + file logging ----
        print(
            f"Epoch [{epoch}/{epochs}] | "
            f"Train Loss: {train_loss_epoch:.6f} | "
            f"Val Loss: {val_loss_epoch:.6f}"
        )

        with open(os.path.join(log_dir, "loss.txt"), "a") as f:
            f.write(f"{epoch},{train_loss_epoch},{val_loss_epoch}\n")

        # ---- wandb logging (per epoch) ----
        if wandb.run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss_epoch": train_loss_epoch,
                    "val_loss_epoch": val_loss_epoch,
                },
                step=epoch,
            )

        # ---- Checkpoint ----
        if ((epoch + 1) % ckpt_every == 0) or (epoch == epochs - 1):
            save_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                save_path,
            )
            print(f"[CKPT] Saved: {save_path}")

    print("Training finished.")


if __name__ == "__main__":
    args = parse_args()
    model = build_model(args.model)
    wandb_init_kwargs = dict(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=vars(args),
    )
    if args.wandb_mode is not None:
        wandb_init_kwargs["mode"] = args.wandb_mode  # "online"/"offline"/"disabled"

    wandb.init(**wandb_init_kwargs)

    # -------------------------
    # Train
    # -------------------------
    train(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dataset_dir=args.dataset_dir,
        log_dir=args.log_dir,
        ckpt_dir=args.ckpt_dir,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        val_every=args.val_every,
        device=args.device,
        rollout_steps=args.rollout_steps,
        num_workers=args.num_workers,
        ckpt_path=args.ckpt_path,
    )

    wandb.finish()
