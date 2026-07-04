# 03 — Data Preprocessing

## Objective
Convert the raw KW51 `.mat` files into a clean, analysis-ready format: one consolidated index of train-passage events, with acceleration and strain time series extracted and stored efficiently, environmental conditions attached, and missing data explicitly flagged rather than silently dropped or imputed at this stage.

## Scope
- Applies to the KW51 `traindata_*` files only (Z24 is already in a preprocessed, analysis-ready format from its source and does not require this stage).
- This stage does not perform feature engineering (rainflow counting, modal analysis, etc.) — that belongs to later stages (Physics Model / PINN Model). This stage only produces clean, structured data that those stages consume.

---

## 1. Verified Raw Data Structure

Each `traindata_YYYYMMDD_HHMMSS.mat` file corresponds to a single train-passage event and contains three MATLAB structs:

### `predat_a` — acceleration
| Field | Type | Shape | Description |
|---|---|---|---|
| `fs` | float | scalar | Sampling rate: 825.81 Hz (verified; may vary slightly file to file — do not hardcode, read from each file) |
| `sdn` | array | (N,) | MATLAB serial date number timestamps, one per sample |
| `labels` | array | (12,) | Channel names: `aBD11Az, aBD17Ay, aBD17Az, aBD17Cz, aBD23Ay, aBD23Az, aAR0910Ay, aAR0910Cy, aAR1516Ay, aAR1516Cy, aAR2122Ay, aAR2122Cy` — `BD` = bridge deck, `AR` = arch, trailing letter = measurement axis/position |
| `tdata` | array | (N, 12) | Acceleration time series, one column per channel, units m/s² |

### `predat_sg` — strain gauge
| Field | Type | Shape | Description |
|---|---|---|---|
| `fs` | float | scalar | Same sampling rate as acceleration |
| `sdn` | array | (N,) | Timestamps |
| `labels` | array | (16,) | Channels 1–12: bridge deck/diagonal strain (`sgBD*`, `sgDI*`); channels 13–16: rail strain (`sgRA*`) — this split matters later, since rail strain and structural strain have different physical meaning and should not be pooled |
| `tdata` | array | (N, 16) | Strain time series, dimensionless (microstrain-equivalent) |

### `predat_env` — environmental conditions
| Field | Type | Shape | Description |
|---|---|---|---|
| `sdn` | float | scalar | Single timestamp for this snapshot (not a time series) |
| `labels` | array | (11,) | `tBD31A` (bridge steel temp), `rhBD31A` (bridge humidity), `tVL` (air temp), `rhVL` (air humidity), `vpVL` (vapor pressure), `grVL` (global radiation), `drVL` (diffuse radiation), `dnrVL` (direct normal radiation), `raVL` (rainfall), `wsVL` (wind speed), `wdVL` (wind direction) |
| `data` | array | (11,) | Corresponding values. **Confirmed to contain real NaNs** (e.g., `wsVL`/`wdVL` were NaN in observed samples) — these are genuine sensor gaps, not a loading error, and must be preserved as NaN, not zero-filled, at this stage |

### `predat_d` — displacement (optional)
**Confirmed absent in all 50 files of October 2018.** Per the original authors' plotting code, this struct only exists in files from later in the monitoring campaign (after displacement sensors were installed). All preprocessing code must check for its presence (`if 'predat_d' in data:`) rather than assume it exists.

---

## 2. Preprocessing Design Decisions

1. **Do not merge months into one giant array in memory.** Process month by month, write one output file per month, to keep memory usage bounded on this instance (16 GB RAM, no GPU headroom concerns here but still finite).
2. **Preserve NaNs, do not impute at this stage.** Imputation strategy (if any) is a modeling decision that belongs later, and different downstream models (PINN vs. baseline) may want different imputation strategies — preprocessing should not make that choice for them.
3. **Keep acceleration and strain aligned on their shared time base**, but store environmental data separately (it's a single value per event, not a time series) rather than broadcasting it across every timestep, which would be misleading (implying it was measured continuously when it wasn't).
4. **Record displacement availability explicitly** as a boolean flag per event, rather than silently omitting the column or filling it with a value that could be misread as real data.
5. **Output format: one `.npz` file per train-passage event** (compact, preserves exact array shapes and dtypes, native to NumPy/PyTorch workflows) **plus one `.csv` index file per month** listing every event with its metadata (timestamp, sampling rate, channel counts, displacement availability, environmental snapshot values). The index is what you and any other researcher will actually browse; the `.npz` files are what the models load.

---

## 3. Directory Structure Produced by This Stage

```
data/processed/kw51/
├── 201810/
│   ├── index.csv
│   └── events/
│       ├── traindata_20181002_160753.npz
│       ├── traindata_20181003_070820.npz
│       └── ...
├── 201811/
│   ├── index.csv
│   └── events/
└── ...
```

Each `index.csv` has one row per event with columns:
`event_id, timestamp, fs_accel, fs_strain, n_accel_channels, n_strain_channels, has_displacement, tBD31A, rhBD31A, tVL, rhVL, vpVL, grVL, drVL, dnrVL, raVL, wsVL, wdVL`

Each `.npz` file contains: `accel` (N×12), `strain` (N×16), `accel_labels`, `strain_labels`, `timestamps`, and `displacement` (N×2, or absent if not available for that event).

---

## 4. Preprocessing Script

Create `src/preprocess_kw51.py`:

```python
"""
Preprocess raw KW51 traindata .mat files into per-event .npz files
plus a per-month CSV index.

Usage:
    python3 src/preprocess_kw51.py --month 201810
    python3 src/preprocess_kw51.py --all
"""

import argparse
import csv
import glob
import os

import numpy as np
import scipy.io as sio

RAW_ROOT = "data/raw/kw51"
PROCESSED_ROOT = "data/processed/kw51"

ENV_LABELS = [
    "tBD31A", "rhBD31A", "tVL", "rhVL", "vpVL",
    "grVL", "drVL", "dnrVL", "raVL", "wsVL", "wdVL",
]

ALL_MONTHS = [
    "201810", "201811", "201812", "201901", "201902", "201903",
    "201904", "201905", "201906", "201907", "201908", "201909",
    "201910", "201911", "201912", "202001",
]


def find_month_files(month: str):
    pattern = os.path.join(RAW_ROOT, f"traindata_{month}", f"traindata_{month}", "*.mat")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No .mat files found for month {month} at expected path: {pattern}. "
            "Confirm the raw data was downloaded and extracted per docs/02_README_Data_Collection.md."
        )
    return files


def process_file(filepath: str, events_dir: str):
    data = sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)

    accel = data["predat_a"]
    strain = data["predat_sg"]
    env = data["predat_env"]

    event_id = os.path.splitext(os.path.basename(filepath))[0]

    output = {
        "accel": accel.tdata,
        "accel_labels": np.array(list(accel.labels)),
        "strain": strain.tdata,
        "strain_labels": np.array(list(strain.labels)),
        "timestamps": accel.sdn,
        "fs_accel": accel.fs,
        "fs_strain": strain.fs,
    }

    has_displacement = "predat_d" in data
    if has_displacement:
        disp = data["predat_d"]
        output["displacement"] = disp.tdata
        output["displacement_labels"] = np.array(list(disp.labels))

    npz_path = os.path.join(events_dir, f"{event_id}.npz")
    np.savez_compressed(npz_path, **output)

    env_values = dict(zip(list(env.labels), list(env.data)))

    row = {
        "event_id": event_id,
        "timestamp": event_id.split("traindata_")[-1],
        "fs_accel": accel.fs,
        "fs_strain": strain.fs,
        "n_accel_channels": accel.tdata.shape[1],
        "n_strain_channels": strain.tdata.shape[1],
        "has_displacement": has_displacement,
    }
    for label in ENV_LABELS:
        row[label] = env_values.get(label, np.nan)

    return row


def process_month(month: str):
    month_dir = os.path.join(PROCESSED_ROOT, month)
    events_dir = os.path.join(month_dir, "events")
    os.makedirs(events_dir, exist_ok=True)

    files = find_month_files(month)
    rows = []
    failed = []

    for filepath in files:
        try:
            row = process_file(filepath, events_dir)
            rows.append(row)
        except Exception as exc:
            failed.append((filepath, str(exc)))

    index_path = os.path.join(month_dir, "index.csv")
    if rows:
        fieldnames = list(rows[0].keys())
        with open(index_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"[{month}] processed {len(rows)}/{len(files)} files successfully")
    if failed:
        print(f"[{month}] FAILED files:")
        for fp, err in failed:
            print(f"  {fp}: {err}")

    return len(rows), len(files), failed


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--month", type=str, help="Process a single month, e.g. 201810")
    group.add_argument("--all", action="store_true", help="Process all 16 months")
    args = parser.parse_args()

    months = ALL_MONTHS if args.all else [args.month]

    total_ok, total_files = 0, 0
    all_failed = []
    for month in months:
        ok, total, failed = process_month(month)
        total_ok += ok
        total_files += total
        all_failed.extend(failed)

    print(f"\nSummary: {total_ok}/{total_files} files processed successfully across {len(months)} month(s)")
    if all_failed:
        print(f"{len(all_failed)} file(s) failed — review the errors above before proceeding.")


if __name__ == "__main__":
    main()
```

### Run it

Test on one month first:
```bash
cd ~/bridge-digital-twin
source venv/bin/activate
python3 src/preprocess_kw51.py --month 201810
```

Confirm the summary line reports 50/50 files processed with no failures, then inspect the output:
```bash
head -5 data/processed/kw51/201810/index.csv
ls data/processed/kw51/201810/events/ | head -5
python3 -c "
import numpy as np
d = np.load('data/processed/kw51/201810/events/traindata_20181002_160753.npz')
print(list(d.keys()))
print('accel shape:', d['accel'].shape)
print('strain shape:', d['strain'].shape)
print('has displacement key:', 'displacement' in d)
"
```

Once the single-month test is verified correct, process everything:
```bash
python3 src/preprocess_kw51.py --all
```

This will take some time (16 months × ~50 files each ≈ 800 files) — let it run to completion and check the final summary line for any failures before proceeding.

---

## 5. Commit the Code (not the processed data)

```bash
git add src/preprocess_kw51.py docs/03_README_Data_Preprocessing.md
git commit -m "Add KW51 preprocessing script and documentation"
git push
```

Confirm `data/processed/` is not tracked:
```bash
git status
```
(should not list any files under `data/processed/`)

---

## Validation Checklist Before Moving to Phase 4

- [ ] `src/preprocess_kw51.py` runs on a single test month (`201810`) with 0 failed files
- [ ] `index.csv` for that month has 50 rows and all expected columns, including environmental columns with visible NaNs where sensors were not reporting
- [ ] A sample `.npz` file loads correctly and contains `accel` (N×12) and `strain` (N×16) with matching labels arrays
- [ ] Confirmed at least one event's `.npz` file correctly has no `displacement` key (matches the verified absence of `predat_d` in October 2018)
- [ ] Full run across all 16 months completes with a final summary reporting failed-file count (0 expected, but if any files fail, they must be individually investigated, not ignored)
- [ ] `git status` confirms `data/processed/` remains untracked
- [ ] Code and this documentation committed and pushed
