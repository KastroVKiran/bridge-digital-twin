# src/train_baseline.py
"""
Baseline (data-loss-only) training loop for the strain reconstruction model.
No physics constraint -- this is the ablation baseline AND the pipeline
correctness check before the physics-informed version is built on top of it.

Usage:
    # Quick pipeline test on a handful of events first:
    python3 src/train_baseline.py --max-events 8 --epochs 2

    # Full training run:
    python3 src/train_baseline.py --epochs 20
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader, Subset

from pinn_dataset import KW51StrainReconstructionDataset
from pinn_model import StrainReconstructionNet

CHECKPOINT_DIR = "checkpoints/baseline"


def make_loader(split: str, batch_size: int, max_events: int = None) -> DataLoader:
    dataset = KW51StrainReconstructionDataset(split)
    if max_events is not None:
        dataset = Subset(dataset, list(range(min(max_events, len(dataset)))))
    shuffle = split == "train"
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def run_epoch(model, loader, optimizer, criterion, train: bool):
    model.train(mode=train)
    total_loss = 0.0
    n_batches = 0

    for accel, strain, event_ids in loader:
        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            pred = model(accel)
            loss = criterion(pred, strain)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-events", type=int, default=None,
                         help="Limit events per split, for quick pipeline testing")
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_loader = make_loader("train", args.batch_size, args.max_events)
    val_loader = make_loader("val", args.batch_size, args.max_events)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = StrainReconstructionNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, criterion, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, train=False)
        elapsed = time.time() - start

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.6e} | "
              f"val_loss={val_loss:.6e} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
            }, checkpoint_path)
            print(f"  -> New best model saved (val_loss={val_loss:.6e})")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.6e}")
    print(f"Best model checkpoint: {os.path.join(CHECKPOINT_DIR, 'best_model.pt')}")


if __name__ == "__main__":
    main()
