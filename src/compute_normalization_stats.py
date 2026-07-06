# src/compute_normalization_stats.py
"""
Compute per-channel mean/std for acceleration (input) and strain (target)
channels, using ONLY the training split, to avoid val/test leakage.
Saves results to data/processed/kw51/normalization_stats.npz.

Usage:
    python3 src/compute_normalization_stats.py
"""

import numpy as np
import os

from pinn_dataset import (
    KW51StrainReconstructionDataset,
    INPUT_ACCEL_CHANNELS,
    TARGET_STRAIN_CHANNELS,
)

OUTPUT_PATH = "data/processed/kw51/normalization_stats.npz"


def main():
    train_dataset = KW51StrainReconstructionDataset("train", normalize=False)
    print(f"Computing normalization stats from {len(train_dataset)} training events...")

    accel_sum = np.zeros(len(INPUT_ACCEL_CHANNELS))
    accel_sumsq = np.zeros(len(INPUT_ACCEL_CHANNELS))
    strain_sum = np.zeros(len(TARGET_STRAIN_CHANNELS))
    strain_sumsq = np.zeros(len(TARGET_STRAIN_CHANNELS))
    n_samples = 0

    for i in range(len(train_dataset)):
        accel, strain, _ = train_dataset[i]
        accel_np = accel.numpy()
        strain_np = strain.numpy()

        accel_sum += accel_np.sum(axis=0)
        accel_sumsq += (accel_np ** 2).sum(axis=0)
        strain_sum += strain_np.sum(axis=0)
        strain_sumsq += (strain_np ** 2).sum(axis=0)
        n_samples += accel_np.shape[0]

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(train_dataset)} events...")

    accel_mean = accel_sum / n_samples
    accel_std = np.sqrt(accel_sumsq / n_samples - accel_mean ** 2)
    strain_mean = strain_sum / n_samples
    strain_std = np.sqrt(strain_sumsq / n_samples - strain_mean ** 2)

    np.savez(
        OUTPUT_PATH,
        accel_mean=accel_mean, accel_std=accel_std,
        strain_mean=strain_mean, strain_std=strain_std,
    )
    print(f"\nSaved normalization stats to {OUTPUT_PATH}")
    print("accel_mean:", accel_mean)
    print("accel_std:", accel_std)
    print("strain_mean:", strain_mean)
    print("strain_std:", strain_std)


if __name__ == "__main__":
    main()
