import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader
from pathlib import Path
import wandb

from unet.unet_model import UNetFiLM
from utils.dataset import CrackDataset
from utils.loss_custom import masked_mse_loss_weighted


def train_model(model, device, train_dir, val_dir, dir_checkpoint,
                epochs=20, batch_size=16, learning_rate=1e-5):

    # ---------------------------
    # Initialize wandb
    # ---------------------------
    wandb.init(
        project="training_w_fem",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "model": "UNetFiLM",
        }
    )

    train_dataset = CrackDataset(train_dir)
    val_dataset   = CrackDataset(val_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):

        # ---------------------------
        # Training
        # ---------------------------
        model.train()
        train_loss = 0

        for batch in train_loader:

            images = batch["image"].to(device)
            masks  = batch["mask"].to(device)
            lc     = batch["lc"].to(device)

            nan_mask = images[:, 9]

            pred = model(images, lc)

            loss = masked_mse_loss_weighted(
                pred.squeeze(1),
                masks.squeeze(1),
                nan_mask
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---------------------------
        # Validation
        # ---------------------------
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:

                images = batch["image"].to(device)
                masks  = batch["mask"].to(device)
                lc     = batch["lc"].to(device)

                nan_mask = images[:, 9]

                pred = model(images, lc)

                loss = masked_mse_loss_weighted(
                    pred.squeeze(1),
                    masks.squeeze(1),
                    nan_mask
                )

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # ---------------------------
        # Log to wandb
        # ---------------------------
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"]
        })

        # ---------------------------
        # Save checkpoint
        # ---------------------------
        dir_checkpoint.mkdir(parents=True, exist_ok=True)
        ckpt_path = dir_checkpoint / f"checkpoint_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt_path)

    wandb.finish()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-dir", type=Path, required=True,
                        help="Path to training dataset")

    parser.add_argument("--val-dir", type=Path, required=True,
                        help="Path to validation dataset")

    parser.add_argument("--checkpoint-dir", type=Path, default=Path("./checkpoints"),
                        help="Directory to save checkpoints")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetFiLM(n_channels=16, n_classes=1)
    model = model.to(device)

    train_model(
        model=model,
        device=device,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        dir_checkpoint=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )