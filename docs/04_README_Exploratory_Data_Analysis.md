# 04 — Exploratory Data Analysis

## Objective
Characterize the processed KW51 dataset before any modeling begins: sensor missingness patterns across the full campaign, signal amplitude ranges, the before/during/after-retrofit class balance, and any anomalies that would undermine later modeling if left undiscovered. This stage produces understanding and figures, not model-ready features — feature engineering belongs to Phase 5 (Physics Model).

## Scope
- Applies to the full processed KW51 dataset (899 events, 16 months).
- Z24 EDA is deferred until Z24 is actually used (Phase 9, generalization testing) — no need to analyze it now.

---

## 1. Known Facts Going Into This Stage (established in Phase 3)

- 899 total events, Oct 2018 – Jan 2020, variable per-event sequence length (train passages differ in duration).
- Retrofit period boundaries: 15 May 2019 – 27 Sep 2019.
- **Class split for before/during/after retrofit: 438 / 248 / 213 events.** This is your primary supervised label for later validation (Phase 9) — a structural state classifier or an anomaly-detection baseline should be checked against this split. Note the "during" class includes the retrofitting work itself, not just a transition point — signals here may reflect construction activity/temporary bracing, not only bridge condition, so treat "during" cautiously as a label rather than assuming it's a clean intermediate structural state.
- Environmental sensor missingness (full campaign): `tBD31A`/`rhBD31A` ~10.5%, `tVL`/`rhVL`/`vpVL` ~0.6%, `grVL`/`drVL`/`dnrVL`/`raVL` ~7.3%, `wsVL`/`wdVL` ~10.1%.
- Displacement present in 172/899 events (~19%), confirmed absent in early months.
- Sample strain amplitude sanity-checked: ~1.8×10⁻⁶ mean, up to 2.2×10⁻⁵ peak on channel `sgBD1011A` — physically reasonable strain magnitudes for train-induced bridge loading (order of 1–20 microstrain).

This stage exists to check whether these single-sample observations hold up across the full dataset, and to surface anything they don't.

---

## 2. EDA Tasks

### 2.1 Missingness over time
Check whether environmental sensor missingness is randomly distributed or concentrated in specific months/periods (e.g., a sensor that was broken for a stretch, not randomly dropping readings). This matters because random missingness can be handled with simple imputation later; concentrated missingness (e.g., "the wind sensor was dead for 3 months") means that variable may be unusable for part of the study period and any model using it must account for that explicitly.

### 2.2 Signal amplitude ranges across all channels, not just one
Verify that every acceleration and strain channel has physically sensible amplitude ranges across the full dataset, not just the one channel spot-checked in Phase 3. A channel with a suspiciously flat signal (near-zero variance) likely indicates a disconnected or faulty sensor for that period — this needs to be known now, not discovered mid-training later when a model mysteriously fails to learn from one input channel.

### 2.3 Sequence length distribution
Since event durations vary, quantify the distribution (min/max/median length) — this directly determines the windowing/padding strategy needed in Phase 5/6 model input pipelines.

### 2.4 Before/during/after class characteristics
Compare basic statistics (mean, std, dominant frequency content via FFT) of acceleration and strain signals across the three retrofit-state classes. This is a sanity check, not the final analysis — if the three classes are indistinguishable even in basic statistics, that's an early warning that damage/retrofit detection will be a harder problem than assumed, and worth knowing before investing in PINN development.

### 2.5 Environmental correlation with structural response
Check whether temperature (a known confound in SHM — structural stiffness and natural frequencies shift with temperature) correlates with strain/acceleration amplitude in this dataset. If it does, later modeling stages need to account for temperature as a covariate, not treat all amplitude variation as damage-related.

---

## 3. EDA Script

Create `src/eda_kw51.py`. This produces figures under `results/figures/eda/` and a summary statistics CSV under `results/tables/`.

```python
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
```

### Run it
```bash
cd ~/bridge-digital-twin
source venv/bin/activate
python3 src/eda_kw51.py
```

This iterates over all 899 events and loads each `.npz` file, so expect it to take a few minutes on a `t3.xlarge` — that is normal, not a hang.

---

## 4. What To Do With the Output

1. Open `results/figures/eda/missingness_over_time.png` — confirm whether missingness is scattered or concentrated in blocks of time. Record the finding in `data/DATASET_METADATA.md`.
2. Check the console output for any "WARNING: channels with unusually low variance" — if any channel appears, inspect it specifically before proceeding, since it may need to be excluded from modeling.
3. Note the sequence length min/median/max from `eda_summary.csv` — this number is required input for Phase 5/6 design (windowing strategy).
4. Note the temperature-strain correlation value — if it's meaningfully non-zero, this confirms temperature must be included as a covariate in the physics-informed model (Phase 5), not treated as noise.
5. Confirm the retrofit class balance figure matches the known counts (438/248/213) as a sanity check that the script itself is correct.

---

## 5. Commit

```bash
git add src/eda_kw51.py docs/04_README_Exploratory_Data_Analysis.md
git commit -m "Add EDA script and documentation for KW51 dataset"
git push
```

Figures and tables under `results/` are small (PNGs and CSVs) and can be committed to Git — unlike raw/processed data, these are final analysis artifacts meant to be version-controlled and eventually referenced in the manuscript:
```bash
git add results/figures/eda/ results/tables/
git commit -m "Add Phase 4 EDA figures and summary tables"
git push
```

---

## Validation Checklist Before Moving to Phase 5

- [ ] `src/eda_kw51.py` runs to completion on all 899 events without errors
- [ ] `eda_summary.csv` confirms before/during/after counts match 438/248/213
- [ ] No unexpected "low variance" channel warnings, or any that appear have been individually investigated and a decision recorded (exclude / keep / investigate further)
- [ ] Sequence length min/median/max recorded and understood
- [ ] Missingness-over-time pattern (scattered vs. concentrated) recorded in `data/DATASET_METADATA.md`
- [ ] Temperature-strain correlation value recorded, with a decision on whether temperature will be included as a covariate in Phase 5
- [ ] All figures, tables, code, and this README committed and pushed
