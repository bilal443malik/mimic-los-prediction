# Data Directory

This folder documents where local MIMIC-IV files should be placed.

## Dataset Citation
- Johnson, Pollard, et al. **MIMIC-IV (version 3.1)**. PhysioNet.
- Link: https://physionet.org/content/mimiciv/3.1/

## Expected Local Layout
```text
data/
├── raw/
│   └── mimiciv/
│       ├── hosp/
│       │   ├── admissions.csv.gz
│       │   ├── patients.csv.gz
│       │   └── ...
│       └── icu/
│           └── ...
├── interim/
└── README.md
```

## Important
- Do not commit patient-level data to version control.
- Access to MIMIC-IV requires credentialed PhysioNet access and data use compliance/training.
- Module defaults may assume CSV extracts in `data/raw/mimiciv/hosp/`.
- The local demo subset (for example MIMIC-IV Demo 2.2) is useful for workflow checks, but may differ from full MIMIC-IV distributions and model performance.

## Minimum Files Needed for Initial LOS Work
- `admissions.csv.gz`
- `patients.csv.gz`
