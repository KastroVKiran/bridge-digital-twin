"""
Classical fatigue-physics model: strain-to-stress conversion, rainflow cycle
counting, S-N curve fatigue life, and Miner's rule cumulative damage.

This module will be reused (not reimplemented) inside the Phase 6 PINN as the
physics-constraint loss term, so correctness here matters directly for later
stages.

Usage:
    python3 src/fatigue_physics.py
"""

import glob
import os

import numpy as np
import pandas as pd
import rainflow

PROCESSED_ROOT = "data/processed/kw51"
RESULTS_TABLES = "results/tables"

E_STEEL_MPA = 210_000.0  # Young's modulus, MPa

# S-N curve detail categories per EN 1993-1-9, used as a sensitivity sweep
# (Delta_sigma_C in MPa at 2e6 cycles), slope m=3 below the constant-amplitude
# fatigue limit -- standard for welded steel details.
DETAIL_CATEGORIES = {"category_56": 56.0, "category_71": 71.0, "category_90": 90.0}
SN_SLOPE_M = 3
SN_REFERENCE_CYCLES = 2e6

DECK_STRAIN_CHANNELS = [
    "sgBD1011A", "sgBD1415A", "sgBD1718A", "sgBD1718C",
    "sgBD2324AB", "sgBD2324AT", "sgBD2324C", "sgBD2728A",
]
DIAGONAL_STRAIN_CHANNELS = ["sgDI20ALB", "sgDI20ALL", "sgDI23ALB", "sgDI23ALL"]


def strain_to_stress_mpa(strain: np.ndarray) -> np.ndarray:
    """Convert dimensionless strain to stress in MPa via Hooke's law."""
    return E_STEEL_MPA * strain


def fatigue_life_cycles(stress_range_mpa: float, detail_category_mpa: float) -> float:
    """N per EN 1993-1-9 S-N curve. Returns inf for zero stress range."""
    if stress_range_mpa <= 0:
        return np.inf
    return SN_REFERENCE_CYCLES * (detail_category_mpa / stress_range_mpa) ** SN_SLOPE_M


def compute_event_damage(strain_series: np.ndarray, detail_category_mpa: float) -> dict:
    """
    Run rainflow counting on one channel's strain time series for one event,
    convert to stress, and compute Miner's rule damage contribution.
    """
    stress_series = strain_to_stress_mpa(strain_series)

    total_damage = 0.0
    n_cycles = 0
    max_range = 0.0

    for rng, mean, count, i_start, i_end in rainflow.extract_cycles(stress_series):
        n_cycles += 1
        max_range = max(max_range, rng)
        N = fatigue_life_cycles(rng, detail_category_mpa)
        if np.isfinite(N) and N > 0:
            total_damage += count / N

    return {
        "damage": total_damage,
        "n_cycles": n_cycles,
        "max_stress_range_mpa": max_range,
    }


def process_channel_set(channel_names: list, label: str):
    files = sorted(glob.glob(os.path.join(PROCESSED_ROOT, "*", "index.csv")))
    full_index = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    full_index["dt"] = pd.to_datetime(full_index["timestamp"], format="%Y%m%d_%H%M%S")

    rows = []
    for _, row in full_index.iterrows():
        month = row["timestamp"][:6]
        npz_path = os.path.join(PROCESSED_ROOT, month, "events", f"{row['event_id']}.npz")
        if not os.path.exists(npz_path):
            continue
        d = np.load(npz_path)
        strain_labels = list(d["strain_labels"])

        available_channels = [c for c in channel_names if c in strain_labels]
        if not available_channels:
            continue

        event_result = {"event_id": row["event_id"], "timestamp": row["timestamp"], "dt": row["dt"]}
        for detail_name, detail_value in DETAIL_CATEGORIES.items():
            channel_damages = []
            for ch in available_channels:
                idx = strain_labels.index(ch)
                result = compute_event_damage(d["strain"][:, idx], detail_value)
                channel_damages.append(result["damage"])
            event_result[f"damage_{detail_name}"] = float(np.mean(channel_damages))
        event_result["n_channels_used"] = len(available_channels)
        rows.append(event_result)

    df = pd.DataFrame(rows).sort_values("dt")
    for detail_name in DETAIL_CATEGORIES:
        df[f"cumulative_damage_{detail_name}"] = df[f"damage_{detail_name}"].cumsum()

    output_path = os.path.join(RESULTS_TABLES, f"fatigue_damage_{label}.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path} ({len(df)} events, mean channels used per event: {df['n_channels_used'].mean():.1f})")
    return df


def main():
    os.makedirs(RESULTS_TABLES, exist_ok=True)

    print("Processing deck strain channels (primary, full campaign coverage)...")
    deck_df = process_channel_set(DECK_STRAIN_CHANNELS, "deck")

    print("\nProcessing diagonal-connection strain channels (supplementary, pre-retrofit only)...")
    diagonal_df = process_channel_set(DIAGONAL_STRAIN_CHANNELS, "diagonal")

    print("\n--- Deck strain: cumulative damage at end of campaign (category_71) ---")
    print(deck_df["cumulative_damage_category_71"].iloc[-1])

    print("\n--- Diagonal strain: cumulative damage at end of available data (category_71) ---")
    print(diagonal_df["cumulative_damage_category_71"].iloc[-1])


if __name__ == "__main__":
    main()
