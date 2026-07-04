"""
Exploratory data analysis for the processed KW51 dataset.
Produces summary tables and figures; does not modify or select final
modeling features.

Usage:
    python3 src/eda_kw51.py
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")  # no display on the EC2 instance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROCESSED_ROOT = "data/processed/kw51"
FIGURES_DIR = "results/figures/eda"
TABLES_DIR = "results/tables"

ENV_COLS = [
    "tBD31A", "rhBD31A", "tVL", "rhVL", "vpVL",
    "grVL", "drVL", "dnrVL", "raVL", "wsVL", "wdVL",
]

RETROFIT_START = pd.Timestamp("2019-05-15")
RETROFIT_END = pd.Timestamp("2019-09-27")


def load_full_index() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(PROCESSED_ROOT, "*", "index.csv")))
    if not files:
        raise FileNotFoundError("No processed index.csv files found. Run Phase 3 preprocessing first.")
    full = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    full["dt"] = pd.to_datetime(full["timestamp"], format="%Y%m%d_%H%M%S")
    full["retrofit_state"] = pd.cut(
        full["dt"],
        bins=[pd.Timestamp.min, RETROFIT_START, RETROFIT_END, pd.Timestamp.max],
        labels=["before", "during", "after"],
    )
    return full


def plot_missingness_over_time(full: pd.DataFrame):
    monthly = full.set_index("dt")[ENV_COLS].resample("ME").apply(lambda x: x.isna().mean() * 100)
    fig, ax = plt.subplots(figsize=(12, 6))
    monthly.plot(ax=ax)
    ax.set_ylabel("Missing (%)")
    ax.set_title("Environmental sensor missingness by month")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "missingness_over_time.png"), dpi=150)
    plt.close(fig)


def compute_channel_amplitude_stats(full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in full.iterrows():
        month = row["timestamp"][:6]
        npz_path = os.path.join(PROCESSED_ROOT, month, "events", f"{row['event_id']}.npz")
        if not os.path.exists(npz_path):
            continue
        d = np.load(npz_path)
        accel, strain = d["accel"], d["strain"]
        for i, label in enumerate(d["accel_labels"]):
            rows.append({
                "event_id": row["event_id"], "channel": str(label), "type": "accel",
                "min": accel[:, i].min(), "max": accel[:, i].max(),
                "std": accel[:, i].std(),
            })
        for i, label in enumerate(d["strain_labels"]):
            rows.append({
                "event_id": row["event_id"], "channel": str(label), "type": "strain",
                "min": strain[:, i].min(), "max": strain[:, i].max(),
                "std": strain[:, i].std(),
            })
    return pd.DataFrame(rows)


def plot_sequence_length_distribution(full: pd.DataFrame):
    lengths = []
    for _, row in full.iterrows():
        month = row["timestamp"][:6]
        npz_path = os.path.join(PROCESSED_ROOT, month, "events", f"{row['event_id']}.npz")
        if not os.path.exists(npz_path):
            continue
        d = np.load(npz_path)
        lengths.append(d["accel"].shape[0])

    lengths = np.array(lengths)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lengths, bins=40)
    ax.set_xlabel("Sequence length (samples)")
    ax.set_ylabel("Number of events")
    ax.set_title(f"Event length distribution (min={lengths.min()}, median={int(np.median(lengths))}, max={lengths.max()})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "sequence_length_distribution.png"), dpi=150)
    plt.close(fig)

    return {"min_length": int(lengths.min()), "median_length": int(np.median(lengths)), "max_length": int(lengths.max())}


def plot_retrofit_class_balance(full: pd.DataFrame):
    counts = full["retrofit_state"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set_ylabel("Number of events")
    ax.set_title("Events per retrofit state")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "retrofit_class_balance.png"), dpi=150)
    plt.close(fig)


def plot_temperature_vs_strain(full: pd.DataFrame, channel_stats: pd.DataFrame):
    merged = full.merge(
        channel_stats[channel_stats["channel"] == "sgBD1011A"][["event_id", "std"]],
        on="event_id", how="inner",
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(merged["tVL"], merged["std"], alpha=0.5, s=10)
    ax.set_xlabel("Air temperature tVL (deg C)")
    ax.set_ylabel("Strain std (channel sgBD1011A)")
    ax.set_title("Temperature vs. strain variability")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "temperature_vs_strain.png"), dpi=150)
    plt.close(fig)

    correlation = merged[["tVL", "std"]].corr().iloc[0, 1]
    return correlation


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    full = load_full_index()
    print(f"Loaded {len(full)} events")
    print(full["retrofit_state"].value_counts())

    plot_missingness_over_time(full)
    print("Saved missingness_over_time.png")

    channel_stats = compute_channel_amplitude_stats(full)
    channel_stats.to_csv(os.path.join(TABLES_DIR, "channel_amplitude_stats.csv"), index=False)
    print("Saved channel_amplitude_stats.csv")

    flat_channels = channel_stats.groupby("channel")["std"].mean()
    flat_channels = flat_channels[flat_channels < flat_channels.quantile(0.05)]
    if len(flat_channels) > 0:
        print("\nWARNING: channels with unusually low variance (possible sensor issue):")
        print(flat_channels)

    length_stats = plot_sequence_length_distribution(full)
    print("Sequence length stats:", length_stats)

    plot_retrofit_class_balance(full)
    print("Saved retrofit_class_balance.png")

    corr = plot_temperature_vs_strain(full, channel_stats)
    print(f"Temperature vs. strain-std correlation (sgBD1011A): {corr:.3f}")

    summary = {
        "total_events": len(full),
        "before_count": int((full["retrofit_state"] == "before").sum()),
        "during_count": int((full["retrofit_state"] == "during").sum()),
        "after_count": int((full["retrofit_state"] == "after").sum()),
        **length_stats,
        "temp_strain_correlation": corr,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(TABLES_DIR, "eda_summary.csv"), index=False)
    print("\nSaved eda_summary.csv")


if __name__ == "__main__":
    main()
