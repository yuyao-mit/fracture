import os
import sys
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import ChunkedScalarDatasetEfficient
from utils.loss import masked_mse
from models import build_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=str)
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--infer_latent_variable", action="store_true")
    parser.add_argument("--rollout_steps", default=1, type=int)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--input_steps", default=4, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    return parser.parse_args()


def load_model(ckpt_path, input_shape, output_shape, model_name, device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    model = build_model(input_shape=input_shape, output_shape=output_shape, model_name=model_name)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model = model.to(device)
    model.eval()

    epoch = ckpt.get("epoch", -1)
    print(f"[Load] Checkpoint from {ckpt_path} (epoch {epoch})")
    return model


def main():
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[Warn] CUDA not available, switching to CPU.")

    dataset = ChunkedScalarDatasetEfficient(
        folder=args.data_dir,
        input_steps=args.input_steps,
        rollout_steps=args.rollout_steps,
        infer_latent_physics=args.infer_latent_variable,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    sample = dataset[0]
    T_in, C_in, H, W = sample["input_states"].shape
    if args.infer_latent_variable:
        target_shape = sample["target"].shape
        T_out, C_out = 1, target_shape[0]
    else:
        target_shape = sample["target"].shape
        T_out, C_out = target_shape[0], target_shape[1]

    input_shape = (args.batch_size, T_in, C_in, H, W)
    output_shape = (args.batch_size, T_out, C_out, H, W)

    model = load_model(args.ckpt_path, input_shape, output_shape, args.model, device)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Infer]"):
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
            total_loss += loss.item()

            if args.output_dir is not None:
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y.cpu().numpy())

    total_loss /= max(1, len(loader))
    print(f"[Result] MSE Loss: {total_loss:.6f}")

    if args.output_dir is not None:
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        np.save(os.path.join(args.output_dir, "predictions.npy"), preds)
        np.save(os.path.join(args.output_dir, "targets.npy"), targets)
        print(f"[Save] Predictions -> {args.output_dir}/predictions.npy  shape={preds.shape}")
        print(f"[Save] Targets     -> {args.output_dir}/targets.npy       shape={targets.shape}")


if __name__ == "__main__":
    main()