# src/pinn_dataset.py
"""
Dataset loader for PINN strain reconstruction: loads sparse acceleration
channels as input, deck strain channels as target, applies fixed-length
windowing (truncate to 36000 samples per Phase 6 design decision).
"""

import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROCESSED_ROOT = "data/processed/kw51"
WINDOW_LENGTH = 36000

INPUT_ACCEL_CHANNELS = ["aBD11Az", "aBD17Ay", "aBD17Az", "aBD23Ay"]
TARGET_STRAIN_CHANNELS = [
    "sgBD1011A", "sgBD1415A", "sgBD1718A", "sgBD1718C",
    "sgBD2324AB", "sgBD2324AT", "sgBD2324C", "sgBD2728A",
]


def build_event_index():
    files = sorted(glob.glob(os.path.join(PROCESSED_ROOT, "*", "index.csv")))
    full = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    full["dt"] = pd.to_datetime(full["timestamp"], format="%Y%m%d_%H%M%S")
    return full


def filter_events_with_complete_channels(full: pd.DataFrame, required_channels: list) -> pd.DataFrame:
    """
    Drop any event where one of the required channels has NaN data, using the
    Phase 4 channel_amplitude_stats.csv (std is NaN exactly when that channel's
    data was NaN for that event -- see Phase 4 findings). This is a proactive
    safeguard against the exact failure mode found during Phase 6 baseline
    testing: silently training on NaN-containing input channels.
    """
    stats_path = "results/tables/channel_amplitude_stats.csv"
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"{stats_path} not found. This file is produced by Phase 4 (src/eda_kw51.py) "
            "and is required here to identify which events have complete data for the "
            "channels this model uses. Run Phase 4's EDA script first."
        )
    stats = pd.read_csv(stats_path)

    valid_event_ids = None
    for channel in required_channels:
        channel_valid = set(stats[(stats["channel"] == channel) & stats["std"].notna()]["event_id"])
        valid_event_ids = channel_valid if valid_event_ids is None else (valid_event_ids & channel_valid)

    before_count = len(full)
    filtered = full[full["event_id"].isin(valid_event_ids)].reset_index(drop=True)
    excluded = before_count - len(filtered)
    if excluded > 0:
        print(f"Excluded {excluded}/{before_count} events missing one or more required channels "
              f"({required_channels}).")

    return filtered


class KW51StrainReconstructionDataset(Dataset):
    def __init__(self, split: str):
        full = build_event_index()
        full = filter_events_with_complete_channels(
            full, INPUT_ACCEL_CHANNELS + TARGET_STRAIN_CHANNELS
        )

        if split == "train":
            mask = full["dt"] < "2019-09-01"
        elif split == "val":
            mask = (full["dt"] >= "2019-09-01") & (full["dt"] < "2019-10-01")
        elif split == "test":
            mask = full["dt"] >= "2019-10-01"
        else:
            raise ValueError(f"Unknown split: {split}")

        self.events = full[mask].reset_index(drop=True)
        self.split = split

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        row = self.events.iloc[idx]
        month = row["timestamp"][:6]
        npz_path = os.path.join(PROCESSED_ROOT, month, "events", f"{row['event_id']}.npz")
        d = np.load(npz_path)

        accel_labels = list(d["accel_labels"])
        strain_labels = list(d["strain_labels"])

        accel_idx = [accel_labels.index(c) for c in INPUT_ACCEL_CHANNELS]
        strain_idx = [strain_labels.index(c) for c in TARGET_STRAIN_CHANNELS]

        accel = d["accel"][:WINDOW_LENGTH, accel_idx]
        strain = d["strain"][:WINDOW_LENGTH, strain_idx]

        if accel.shape[0] < WINDOW_LENGTH:
            raise ValueError(
                f"Event {row['event_id']} shorter than WINDOW_LENGTH ({accel.shape[0]} < {WINDOW_LENGTH}) "
                "-- this should not happen given Phase 4's confirmed minimum length of 36333; "
                "investigate this specific event before proceeding."
            )

        return (
            torch.tensor(accel, dtype=torch.float32),
            torch.tensor(strain, dtype=torch.float32),
            row["event_id"],
        )


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        ds = KW51StrainReconstructionDataset(split)
        print(f"{split}: {len(ds)} events")
        accel, strain, event_id = ds[0]
        print(f"  sample shapes: accel {accel.shape}, strain {strain.shape}, event_id {event_id}")
