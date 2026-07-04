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
