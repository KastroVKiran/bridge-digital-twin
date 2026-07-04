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

- CRITICAL FINDING: diagonal-connection strain gauges (sgDI20ALB, sgDI20ALL, sgDI23ALB,
  sgDI23ALL) are present ONLY from Nov 2018 to 7 May 2019 (345/899 events), entirely within
  the pre-retrofit period. Zero coverage during or after retrofit (confirmed, not relabeled
  elsewhere). Bridge-deck strain gauges (sgBD*) have full 899/899 coverage across all three
  retrofit states. Research scope decision: primary fatigue analysis will use deck strain
  (sgBD*) for full before/during/after comparison; diagonal-connection strain (sgDI*) will be
  used as a supplementary pre-retrofit-only characterization at the specific location that
  was later retrofitted, not as a before/after comparison.
- aBD23Ay (accel) and sgBD1415A (strain) flagged as lowest-variance within their sensor type,
  but both have near-complete coverage (886/899 and 899/899) and smooth, non-degenerate
  distributions -- genuinely low-signal sensors, not faulty, no exclusion needed.
- Six arch-mounted accelerometers (aAR0910/1516/2122, Ay/Cy) missing in 687/899 events (76%) --
  installed partway through campaign, needs dating before use in any arch-vibration analysis.

## Phase 5 Fatigue Physics Results
- Deck strain cumulative damage (category 71, full campaign): 4.80e-07
- Diagonal strain cumulative damage (category 71, pre-retrofit period only, 345 events): 2.53e-06
- Diagonal per-event damage correctly zero for the 554 events without valid sensor data
  (NaN arrays produce zero rainflow cycles by construction, not a bug)
