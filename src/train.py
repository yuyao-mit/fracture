import os
import sys
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import argparse
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import ChunkedScalarDatasetEfficient
from utils.loss import masked_mse
from models import build_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True, type=str)
    parser.add_argument("--val-dir", required=True, type=str)
    parser.add_argument("--ckpt_dir", required=True, type=str)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--ckpt_path", default=None, type=str)
    parser.add_argument("--infer_latent_variable", action="store_true")
    parser.add_argument("--rollout_steps", default=1, type=int)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--input_steps", default=4, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--ckpt_every", default=10, type=int)
    parser.add_argument("--wandb_project", default="fracture_ilp", type=str)
    parser.add_argument("--wandb_run_name", default=None, type=str)
    parser.add_argument("--wandb_mode", default=None, type=str)
    return parser.parse_args()


def _load_ckpt(model, optimizer, ckpt_path, device):
    if ckpt_path is None:
        return 0
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    print(f"[Resume] Loaded from {ckpt_path}, start_epoch={start_epoch}")
    return start_epoch


def main():
    args = parse_args()

    wandb_kwargs = dict(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )
    if args.wandb_mode is not None:
        wandb_kwargs["mode"] = args.wandb_mode
    wandb.init(**wandb_kwargs)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[Warn] CUDA not available, switching to CPU.")

    train_dataset = ChunkedScalarDatasetEfficient(
        folder=args.train_dir,
        input_steps=args.input_steps,
        rollout_steps=args.rollout_steps,
        infer_latent_physics=args.infer_latent_variable,
    )
    val_dataset = ChunkedScalarDatasetEfficient(
        folder=args.val_dir,
        input_steps=args.input_steps,
        rollout_steps=args.rollout_steps,
        infer_latent_physics=args.infer_latent_variable,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    sample = train_dataset[0]
    T_in, C_in, H, W = sample["input_states"].shape
    if args.infer_latent_variable:
        target_shape = sample["target"].shape   # [C_latent, H, W]
        T_out, C_out = 1, target_shape[0]
    else:
        target_shape = sample["target"].shape   # [rollout_steps, 1, H, W]
        T_out, C_out = target_shape[0], target_shape[1]

    input_shape = (args.batch_size, T_in, C_in, H, W)
    output_shape = (args.batch_size, T_out, C_out, H, W)

    model = build_model(input_shape=input_shape, output_shape=output_shape, model_name=args.model)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    start_epoch = _load_ckpt(model, optimizer, args.ckpt_path, device)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):

        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"[Train] Epoch {epoch}"):
            x = batch["input_states"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            nan_mask = x[:, -1, 9]

            pred = model(x)

            if args.infer_latent_variable:
                pred = pred.squeeze(1)
                mask = nan_mask.unsqueeze(1).expand_as(pred)
            else:
                mask = nan_mask.unsqueeze(1).unsqueeze(2).expand_as(pred)

            loss = masked_mse(pred, y, mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_states"].to(device, non_blocking=True)
                y = batch["target"].to(device, non_blocking=True)
                nan_mask = x[:, -1, 9]

                pred = model(x)

                if args.infer_latent_variable:
                    pred = pred.squeeze(1)
                    mask = nan_mask.unsqueeze(1).expand_as(pred)
                else:
                    mask = nan_mask.unsqueeze(1).unsqueeze(2).expand_as(pred)

                loss = masked_mse(pred, y, mask)
                val_loss += loss.item()

        val_loss /= max(1, len(val_loader))

        print(
            f"Epoch [{epoch}/{args.epochs}] | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, step=epoch)

        if (epoch + 1) % args.ckpt_every == 0 or epoch == args.epochs - 1:
            save_path = os.path.join(args.ckpt_dir, f"epoch_{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, save_path)
            print(f"[CKPT] Saved: {save_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
