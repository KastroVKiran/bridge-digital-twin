# src/evaluate_baseline.py
"""
Evaluate the trained baseline strain-reconstruction model on the held-out
test set: per-channel MSE (on de-normalized, physical-unit strain), plus
fatigue-damage discrepancy computed via the Phase 5 rainflow + Miner's rule
pipeline on de-normalized reconstructed vs. true strain.

Usage:
    python3 src/evaluate_baseline.py
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pinn_dataset import KW51StrainReconstructionDataset, TARGET_STRAIN_CHANNELS, denormalize_strain
from pinn_model import StrainReconstructionNet
from fatigue_physics import compute_event_damage, DETAIL_CATEGORIES

CHECKPOINT_PATH = "checkpoints/baseline/best_model.pt"
RESULTS_TABLES = "results/tables"


def load_model():
    model = StrainReconstructionNet()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.6e}")
    return model


def main():
    os.makedirs(RESULTS_TABLES, exist_ok=True)

    model = load_model()
    test_dataset = KW51StrainReconstructionDataset("test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Evaluating on {len(test_dataset)} test events...")

    per_channel_errors = {ch: [] for ch in TARGET_STRAIN_CHANNELS}
    damage_rows = []

    with torch.no_grad():
        for i, (accel, strain_true, event_ids) in enumerate(test_loader):
            pred = model(accel)

            # De-normalize back to physical strain units before any physical
            # interpretation (MSE reporting or fatigue physics) -- the model
            # operates in normalized space, but rainflow counting and Miner's
            # rule require real physical strain values.
            pred_physical = denormalize_strain(pred.squeeze(0).numpy())
            true_physical = denormalize_strain(strain_true.squeeze(0).numpy())

            # Per-channel MSE, computed on physical-unit values
            mse_per_channel = ((pred_physical - true_physical) ** 2).mean(axis=0)
            for ch_idx, ch_name in enumerate(TARGET_STRAIN_CHANNELS):
                per_channel_errors[ch_name].append(mse_per_channel[ch_idx])

            # Fatigue damage discrepancy (category 71 only, for speed -- the
            # full 3-category sweep is unnecessary here since we're comparing
            # reconstruction quality, not re-deriving the Phase 5 sensitivity result)
            pred_damages, true_damages = [], []
            for ch_idx in range(len(TARGET_STRAIN_CHANNELS)):
                pred_result = compute_event_damage(pred_physical[:, ch_idx], DETAIL_CATEGORIES["category_71"])
                true_result = compute_event_damage(true_physical[:, ch_idx], DETAIL_CATEGORIES["category_71"])
                pred_damages.append(pred_result["damage"])
                true_damages.append(true_result["damage"])

            damage_rows.append({
                "event_id": event_ids[0],
                "damage_pred_mean": float(np.mean(pred_damages)),
                "damage_true_mean": float(np.mean(true_damages)),
            })

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(test_dataset)} events...")

    # Save per-channel MSE summary (physical units)
    mse_summary = {ch: float(np.mean(errs)) for ch, errs in per_channel_errors.items()}
    mse_df = pd.DataFrame([mse_summary])
    mse_df.to_csv(os.path.join(RESULTS_TABLES, "baseline_test_mse_per_channel.csv"), index=False)
    print("\nPer-channel test MSE (physical strain units):")
    print(mse_df.T.rename(columns={0: "mse"}))

    # Save fatigue damage comparison
    damage_df = pd.DataFrame(damage_rows)
    damage_df["damage_error"] = damage_df["damage_pred_mean"] - damage_df["damage_true_mean"]
    damage_df["damage_abs_pct_error"] = (
        (damage_df["damage_error"].abs() / damage_df["damage_true_mean"].replace(0, np.nan)) * 100
    )
    damage_df.to_csv(os.path.join(RESULTS_TABLES, "baseline_test_damage_comparison.csv"), index=False)

    print(f"\nFatigue damage comparison (n={len(damage_df)} test events):")
    print(f"  Mean true damage:  {damage_df['damage_true_mean'].mean():.6e}")
    print(f"  Mean pred damage:  {damage_df['damage_pred_mean'].mean():.6e}")
    print(f"  Mean absolute %% error: {damage_df['damage_abs_pct_error'].mean():.2f}%%")

    print("\nSaved: baseline_test_mse_per_channel.csv, baseline_test_damage_comparison.csv")


if __name__ == "__main__":
    main()
