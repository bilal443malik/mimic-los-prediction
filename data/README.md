# Data Directory

This folder documents where local MIMIC-IV files should be placed.

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
- Access to MIMIC-IV requires credentialed PhysioNet access and data use compliance.
- Module defaults may assume CSV extracts in `data/raw/mimiciv/hosp/`.

## Minimum Files Needed for Initial LOS Work
- `admissions.csv.gz`
- `patients.csv.gz`
