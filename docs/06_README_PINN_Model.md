# 06 — PINN Model (Physics-Informed Strain Reconstruction)

## Objective
Build a neural network that reconstructs full deck strain time series from a **sparse subset of acceleration channels**, constrained by the fatigue physics from Phase 5 (rainflow-consistent stress ranges, Miner's rule damage), so that a bridge instrumented with only accelerometers (cheaper, more common in practice) can still produce reliable fatigue-damage estimates without needing strain gauges everywhere.

**This is the paper's core novelty.** Reconstructing strain from acceleration is not new; embedding fatigue-cumulative-damage consistency as a physics constraint in that reconstruction, validated against classical rainflow/Miner's-rule physics from real monitoring data, is the specific contribution.

## Scope
- Input: a sparse subset of the 12 acceleration channels (simulating a cheaper, partially-instrumented bridge).
- Output: reconstructed deck strain (`sgBD*`, 8 channels — the full-coverage channel set from Phase 4/5).
- Physics constraint: reconstructed strain, when passed through the exact Phase 5 rainflow + Miner's rule pipeline, should produce cumulative damage consistent with the damage computed from real strain (Phase 5's `fatigue_damage_deck.csv` is therefore your ground truth for this constraint, not a discarded intermediate result).
- Out of scope: diagonal-connection channels (too sparse in time — pre-retrofit only — to serve as a training target for a model meant to generalize across the full campaign). Diagonal data is reserved for a focused pre-retrofit case study in a later results section, not for PINN training here.

---

## 1. Problem Formulation

**Input:** a sparse subset of acceleration channels, e.g. 4 of the 12 available (`aBD11Az`, `aBD17Ay`, `aAR0910Ay`, `aAR2122Ay` — one from each structural zone: deck, deck, arch, arch) — chosen to simulate a realistically minimal instrumentation scenario, not the full 12-channel array a real cost-constrained deployment wouldn't have.

**Output:** all 8 deck strain channels (`sgBD*`), full time series.

**Two loss terms:**
1. **Data loss** — standard supervised reconstruction loss (MSE) between predicted and true strain, using the real strain gauge data as ground truth during training (this is standard for any PINN — physics constraints supplement, not replace, available data).
2. **Physics loss** — the reconstructed strain, run through the Phase 5 rainflow + Miner's rule pipeline, should yield a cumulative damage value close to the true damage computed from real strain (Phase 5's results). This is what makes the model "physics-informed": it's penalized not just for point-wise strain error, but for getting the *fatigue-relevant* cycle content of the signal wrong, even if raw waveform error looks small.

```
L_total = w_data * L_data + w_physics * L_physics
```
where `L_data = MSE(strain_pred, strain_true)` and `L_physics = (D_pred - D_true)^2`, with `D_pred` and `D_true` computed via the same Phase 5 fatigue pipeline (differentiable rainflow counting is not standard — see Section 3 for how this is handled).

---

## 2. Honest Technical Challenge: Rainflow Counting Is Not Differentiable

Standard rainflow counting (peak-valley extraction with a stack-based algorithm) is not a differentiable operation — you cannot backpropagate through it directly, which is a real obstacle to using it as-is inside a neural network loss function. This project addresses it as follows, and this should be stated plainly in the manuscript as a methodological design choice, not hidden:

**Approach: physics loss computed on a fixed cadence, not every training step.** Rather than attempting a differentiable rainflow approximation (an active but immature research area), this project computes the data loss every training step (fully differentiable, drives most of the learning), and computes the physics loss periodically (e.g., every 10 steps or once per epoch) by running the actual non-differentiable rainflow pipeline on the current model's reconstructed strain, then using the **resulting damage discrepancy as a scalar penalty added to the loss**, with gradients approximated via a straight-through estimator on the stress-range distribution (i.e., the gradient flows through the stress-range values that feed into the rainflow-derived damage number, treating the cycle-counting/binning step itself as constant during backpropagation). This is a legitimate, published pattern for embedding non-differentiable domain algorithms into neural network training (similar in spirit to how non-differentiable rendering or discrete counting operations are handled elsewhere in physics-informed ML), and is meaningfully different from — and more honest than — simply pretending rainflow counting is smooth.

**Simpler fallback, if the above proves unstable in practice (decide during implementation, not now):** approximate the physics loss with a differentiable proxy — e.g., penalize the discrepancy in the *stress range distribution's higher moments* (variance, kurtosis) between predicted and true strain, which correlates strongly with Miner's-rule damage without requiring literal rainflow counting in the loop. This is a weaker physics constraint but fully differentiable and simpler to debug. **Recommendation: implement the fixed-cadence approach first (Section 3), since it's the more scientifically honest and defensible version for the paper; fall back to the proxy only if training proves unstable.**

---

## 3. Architecture

**Recommendation: a 1D CNN + LSTM hybrid**, not a Transformer, for this stage. Reasoning: your sequences are long (36k-107k samples per event, per Phase 4), a full Transformer's quadratic attention cost is unnecessary and slow on a CPU-only `t3.xlarge`; a 1D CNN front-end (captures local waveform patterns — impulse events, oscillation frequency) feeding an LSTM (captures longer-range temporal dependency across the passage) is a well-established, computationally lighter architecture for this class of problem, and appropriate given you have no GPU in this project's current setup.

```
Input: (batch, sequence_length, n_input_accel_channels)
  -> 1D Conv layers (kernel size ~ 15-31, capturing oscillation-scale patterns given ~825 Hz sampling)
  -> LSTM (bidirectional, capturing full-passage context)
  -> Fully connected output layer -> (batch, sequence_length, 8)  # 8 deck strain channels
```

## 4. Handling Variable Sequence Length (Phase 4 Finding)

Recall Phase 4: sequence lengths range 36,333 to 107,326 samples, with a long tail. This project's approach: **fixed-length windowing**, not padding to the max length (padding to 107k when the median is 43k would waste enormous compute on padding tokens for most events). Specifically:
- Truncate every event to the first **36,000 samples** (the campaign minimum, ensuring every event contributes a full, real window with zero padding).
- For events longer than 36,000 samples, this discards the tail — acceptable, since the train-passage transient (the fatigue-relevant loading event) occurs at the start of each recording, not in extended tail data.
- State this explicitly as a design decision with its rationale in the manuscript, not as an unexamined default.

## 5. Data Splits

**Temporal split, not random split** — standard and necessary for time-series SHM data to avoid leakage (a randomly-selected test event could be adjacent in time to a training event, artificially inflating performance):
- **Train:** Oct 2018 – Aug 2019 (pre-retrofit + during-retrofit period)
- **Validation:** Sep 2019 (end of during-retrofit period)
- **Test:** Oct 2019 – Jan 2020 (fully post-retrofit, unseen structural state)

This split is itself a meaningful test of generalization: can a model trained before/during retrofit correctly reconstruct strain (and therefore damage) in the post-retrofit structural state it never saw? This is a legitimate and interesting result either way — success shows genuine generalization; a measurable performance drop after retrofit is itself a finding worth reporting (the model detecting the structural change indirectly through reconstruction error, which is actually a bonus damage/change-detection result you get for free).

---

## 6. Baseline Model (Required for the Ablation)

Train the identical architecture (same CNN+LSTM) with `w_physics = 0` (data loss only) as the baseline. This isolates the physics constraint's actual contribution — the central empirical claim of the paper — rather than just presenting the physics-informed model's absolute performance with nothing to compare against.

---

## 7. Implementation

Install additional dependencies:
```bash
cd ~/bridge-digital-twin
source venv/bin/activate
pip install scikit-learn
pip freeze > requirements.txt
```

Create `src/pinn_dataset.py` (data loading/windowing), `src/pinn_model.py` (architecture), and `src/train_pinn.py` (training loop with both loss terms). Given the size of this stage, these will be built and tested incrementally — starting with the dataset loader, verified correct, before writing a single line of model code. This mirrors the same discipline used in Phases 1-5: verify each layer works before building the next one on top of it.

### Step 1: Dataset loader — build and verify this first

```python
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

INPUT_ACCEL_CHANNELS = ["aBD11Az", "aBD17Ay", "aAR0910Ay", "aAR2122Ay"]
TARGET_STRAIN_CHANNELS = [
    "sgBD1011A", "sgBD1415A", "sgBD1718A", "sgBD1718C",
    "sgBD2324AB", "sgBD2324AT", "sgBD2324C", "sgBD2728A",
]


def build_event_index():
    files = sorted(glob.glob(os.path.join(PROCESSED_ROOT, "*", "index.csv")))
    full = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    full["dt"] = pd.to_datetime(full["timestamp"], format="%Y%m%d_%H%M%S")
    return full


class KW51StrainReconstructionDataset(Dataset):
    def __init__(self, split: str):
        full = build_event_index()
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
```

### Run and verify this first, before anything else:
```bash
cd ~/bridge-digital-twin
python3 src/pinn_dataset.py
```

**Do not proceed to the model architecture until this runs cleanly** and the train/val/test counts look reasonable (roughly matching the expected ~730/~40/~130 event split based on the date ranges, though exact numbers depend on which events actually exist per month per Phase 3's real counts).

---

## Next Steps After This Verification

Once `pinn_dataset.py` is confirmed working, the remaining Phase 6 components (model architecture in `pinn_model.py`, training loop with the two-term loss in `train_pinn.py`, and the baseline-vs-physics-informed ablation) will be built in that order — each verified before the next is written. This is intentionally incremental rather than delivered as one large untested block, since a bug in the dataset loader would silently corrupt every result built on top of it.

## Validation Checklist for This Sub-Stage

- [ ] `pinn_dataset.py` runs without errors for all three splits
- [ ] Train/val/test event counts are non-zero and roughly match the expected date-range proportions
- [ ] A sample `accel`/`strain` pair has the expected shapes: `(36000, 4)` and `(36000, 8)`
- [ ] No event raises the "shorter than WINDOW_LENGTH" error (if one does, investigate that specific event before continuing — do not silently skip it)
