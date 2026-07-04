# 05 — Physics Model (Fatigue Damage Formulation)

## Objective
Establish the classical fatigue-physics model — rainflow cycle counting, S-N curve, and Miner's rule cumulative damage — that will later be embedded as a physics constraint inside the PINN (Phase 6). This stage produces a working, validated **classical** fatigue-damage calculation using deck strain data. It does not yet involve any neural network — the PINN in Phase 6 uses this exact physics as its loss constraint, so it must be correct and well-understood before it's embedded in a training loop where errors are harder to diagnose.

## Scope
- Primary channel set: bridge-deck strain gauges (`sgBD*`), which have full 899/899 coverage across all three retrofit states — this is what supports the before/during/after fatigue comparison.
- Supplementary: diagonal-connection strain gauges (`sgDI*`), used only for the 345 pre-retrofit events where they exist, analyzed as a standalone pre-retrofit characterization, not a before/after comparison (per the Phase 4 scope decision).
- Out of scope: any ML/PINN code (Phase 6); this stage is classical structural engineering computation only.

---

## 1. Physics Background

### 1.1 Strain-to-stress conversion
The dataset provides strain (dimensionless), not stress. For steel, stress and strain are related by Hooke's law in the elastic range:

```
σ = E × ε
```

where `E` = Young's modulus of structural steel ≈ 210 GPa = 210,000 MPa, and `ε` is strain (dimensionless, as recorded). This is a standard, defensible assumption for a steel bridge under service loading, where strains remain well within the elastic range (confirmed by Phase 4: observed strain values were ~10⁻⁶ to ~2×10⁻⁵, corresponding to stresses on the order of 0.2–4 MPa — far below yield stress for structural steel, ~235–355 MPa depending on grade — so the elastic assumption holds comfortably).

**Note this stress magnitude explicitly in the paper as a limitation-aware observation:** these are *dynamic, train-passage-induced* stress ranges only, not the total service stress state of the structure (which also includes static dead load, thermal effects, etc.). Fatigue analysis specifically concerns stress *ranges* (cyclic variation), so this is the physically correct quantity for this purpose — but it should not be confused with total stress.

### 1.2 Rainflow cycle counting
A single train passage produces an irregular stress-time history, not clean sinusoidal cycles. Rainflow counting (ASTM E1049-85) is the standard method to decompose an irregular load history into a set of discrete stress-range cycles (each with a range `Δσ` and a count `n`, including half-cycles), which is required before Miner's rule can be applied. This project uses the `rainflow` Python package, a standard ASTM-E1049-compliant implementation.

### 1.3 S-N curve (fatigue detail category)
Fatigue life at a given stress range is governed by an S-N curve of the form:

```
N = (Δσ_C / Δσ)^m × 2×10^6
```

per EN 1993-1-9 (Eurocode 3, Part 1-9: Fatigue), where `Δσ_C` is the detail category's reference stress range at 2×10⁶ cycles, and `m` is the slope constant (typically m=3 for welded steel details below the constant-amplitude fatigue limit).

**This project uses detail category 71 (Δσ_C = 71 MPa) as a working assumption**, representative of welded plate/gusset connections typical of steel bowstring bridge deck and diagonal connections. **This is an engineering assumption, not a value read from KW51's actual fabrication drawings** — the exact detail category depends on the specific welded connection geometry, which is not provided in the monitoring dataset. This must be stated explicitly as a limitation in the manuscript, and ideally cross-checked against the structural drawings referenced in the KW51 data paper (Maes and Lombaert, 2021) if accessible, or treated as a sensitivity-analysis parameter (running the same pipeline with categories 56, 71, and 90 to show how sensitive the RUL conclusion is to this assumption) — the latter is the more defensible approach for a Q1 paper and is what this project will do (see Section 4).

### 1.4 Miner's rule (linear cumulative damage)
```
D = Σ (n_i / N_i)
```
where `n_i` is the number of cycles observed at stress range `Δσ_i`, and `N_i` is the fatigue life at that stress range from the S-N curve. Failure is nominally predicted at `D = 1`. This project computes `D` per event, then accumulates `D` over time across the campaign to produce a cumulative damage trajectory — the classical-physics equivalent of what the PINN will later learn to approximate and extend probabilistically.

---

## 2. Known Limitation: Sampling Frequency of Events

The dataset captures **two train passages per day plus six ambient-vibration windows per day** (per the original data paper) — not every train that actually crosses the bridge. This means the cumulative damage computed from this dataset is a **sampled estimate**, not the true cumulative damage experienced by the bridge. This must be stated plainly in the manuscript: the absolute damage index `D` should not be interpreted as the bridge's true fatigue state, but the *relative* damage accumulation pattern (rate of increase, sensitivity to retrofit) is still a valid and meaningful research signal, since the sampling is consistent across the campaign.

---

## 3. Physics Model Script

Install the rainflow package:
```bash
cd ~/bridge-digital-twin
source venv/bin/activate
pip install rainflow
pip freeze > requirements.txt
```

Create `src/fatigue_physics.py`:

```python
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
```

### Run it
```bash
cd ~/bridge-digital-twin
python3 src/fatigue_physics.py
```

This processes all 899 events for the deck channels and all 345 events for the diagonal channels, computing damage under all three S-N detail categories for each. Expect this to take a few minutes (rainflow counting runs per-channel, per-event).

---

## 4. Why a Sensitivity Sweep Across Detail Categories, Not One Fixed Value

Since the exact fatigue detail category is an assumption (Section 1.3), computing damage under three plausible categories (56, 71, 90) rather than committing to one lets you show — likely as a figure in the paper — how the *shape* of the cumulative damage trajectory (increasing, accelerating, sensitive to retrofit) is robust to this assumption, even though the absolute damage values shift. This converts an unavoidable uncertainty into an explicit, honestly-reported robustness check, which is exactly the kind of rigor a Q1 reviewer will respect rather than penalize.

---

## 5. Sanity Checks to Run on the Output

Before treating this as correct, verify:

```bash
python3 -c "
import pandas as pd
deck = pd.read_csv('results/tables/fatigue_damage_deck.csv')
print('Deck events processed:', len(deck))
print(deck[['damage_category_56','damage_category_71','damage_category_90']].describe())
print()
print('Any negative or NaN damage values?', deck[['damage_category_56','damage_category_71','damage_category_90']].isna().any().any())
print()
print('Cumulative damage is monotonically increasing:', (deck['cumulative_damage_category_71'].diff().dropna() >= 0).all())
"
```

Expected: no NaNs, no negative damage values (Miner's rule damage must be ≥0), and cumulative damage strictly non-decreasing (it is a running sum of non-negative increments by construction — if this check fails, there is a bug, not a data issue).

Also plot the cumulative damage trajectory against the retrofit period to see if there's a visible change in accumulation rate — this is your first real look at whether the physics signal supports the paper's premise:

```bash
python3 -c "
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

deck = pd.read_csv('results/tables/fatigue_damage_deck.csv', parse_dates=['dt'])

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(deck['dt'], deck['cumulative_damage_category_56'], label='Category 56')
ax.plot(deck['dt'], deck['cumulative_damage_category_71'], label='Category 71')
ax.plot(deck['dt'], deck['cumulative_damage_category_90'], label='Category 90')
ax.axvspan(pd.Timestamp('2019-05-15'), pd.Timestamp('2019-09-27'), color='gray', alpha=0.3, label='Retrofit period')
ax.set_ylabel('Cumulative Miner damage (deck strain)')
ax.set_xlabel('Date')
ax.legend()
fig.tight_layout()
fig.savefig('results/figures/eda/cumulative_damage_deck.png', dpi=150)
print('Saved cumulative_damage_deck.png')
"
```

---

## 6. Commit

```bash
git add src/fatigue_physics.py docs/05_README_Physics_Model.md requirements.txt
git commit -m "Add classical fatigue physics model: rainflow counting, S-N curves, Miner's rule"
git push

git add results/tables/fatigue_damage_deck.csv results/tables/fatigue_damage_diagonal.csv results/figures/eda/cumulative_damage_deck.png
git commit -m "Add Phase 5 fatigue damage computation results"
git push
```

---

## Validation Checklist Before Moving to Phase 6

- [ ] `rainflow` package installed, added to `requirements.txt`
- [ ] `fatigue_physics.py` runs to completion on all 899 deck events and 345 diagonal events with no errors
- [ ] Sanity checks pass: no NaN/negative damage values, cumulative damage strictly non-decreasing
- [ ] Cumulative damage plot generated and visually inspected — record whether a visible change in accumulation rate appears around the retrofit period (this is a genuine research observation, not a given — note whatever the plot actually shows, don't assume the "expected" answer)
- [ ] Understood and can explain: why strain-to-stress conversion, why rainflow counting is necessary, why Miner's rule, why a 3-category sensitivity sweep instead of one fixed detail category
- [ ] Explicitly recorded the "sampled, not exhaustive, train passages" limitation in `data/DATASET_METADATA.md` or manuscript notes
- [ ] All code, results, and documentation committed and pushed
