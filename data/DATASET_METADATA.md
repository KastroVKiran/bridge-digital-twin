# Dataset Metadata

## KW51 Railway Bridge Monitoring Data
- Source: Zenodo, DOI 10.5281/zenodo.3745914
- License: CC BY-NC-SA 4.0
- Files obtained: readme.txt, matlab-functions.zip, trackedmodes.zip, traindata_201810.zip through traindata_202001.zip (15 files)
- Files intentionally not obtained: ambient_*.zip (15 files, ~41 GB) — raw ambient vibration data, not required given trackedmodes.zip provides the processed modal-parameter evolution
- Download date: [fill in actual date]
- Verified against official MD5 checksums published on the Zenodo record page

## Z24 Bridge Benchmark (processed)
- Source: Hugging Face, thanglexuan/Z24-dataset-processed
- Original data: Maeck, J. and De Roeck, G. (2003), Mechanical Systems and Signal Processing, 17(1), 127-131
- Access: derived/reshaped dataset, downloaded directly; raw data requested separately from KU Leuven Structural Mechanics Section (https://bwk.kuleuven.be/bwm/z24)
- Download date: [fill in actual date]
- Raw data request status: [pending / granted / not granted — update once known]

## KW51 Preprocessing Summary (Phase 3)
- Total events processed: 899 (across 16 months, Oct 2018 - Jan 2020)
- Environmental sensor missingness: tBD31A/rhBD31A ~10.5%, tVL/rhVL/vpVL ~0.6%,
  grVL/drVL/dnrVL/raVL ~7.3%, wsVL/wdVL ~10.1%
- Displacement data (predat_d) present in 172/899 events (~19%); confirmed absent
  in earlier months (e.g. all of Oct 2018), added partway through the campaign
