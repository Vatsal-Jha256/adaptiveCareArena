# Getting the data

The repository contains no participant data. Both cohorts are public,
de-identified research datasets that you download from their owners and place
under `data/raw/`, which is gitignored. Do not commit or redistribute the
files. Do not attempt to re-identify any participant.

| Cohort | Participants used | Owner's page | Terms |
|---|---|---|---|
| CrossCheck | 64 (of 75 enrolled) | https://www.mh4mh.org/crosscheck | De-identified, approved for public use. No standard reuse license is stated, so treat it as download-only. |
| College Experience Study (CES) | 210 | https://www.kaggle.com/datasets/subigyanepal/college-experience-dataset | CC BY-NC-SA 4.0 per the dataset's Kaggle metadata. Non-commercial. Redistribute no rows. |

The numbers above are what the benchmark ends up with after its eligibility
rules (at least 20 genuine EMA reports per participant); the raw files contain
more people.

## CrossCheck

1. Open the page above and download the three files it links:
   - `CrossCheck_Daily_Data.csv`
   - `CrossCheck_Hourly_Data_v1_08132020.csv`
   - `CrossCheck Data Dictionary Daily and Hou.xlsx` (documentation only)
2. Put them in `data/raw/crosscheck/`:

```
data/raw/crosscheck/
  CrossCheck_Daily_Data.csv
  CrossCheck_Hourly_Data_v1_08132020.csv
  CrossCheck Data Dictionary Daily and Hou.xlsx
```

Schema notes the adapter relies on:

- The daily table has 155 columns keyed by `study_id, eureka_id, day`.
- **EMA is administered every 2-3 days, not daily.** Every row has the `ema_*`
  columns, but only a fraction hold a genuine response. The adapter counts and
  targets only rows with a real response and never forward-fills one into a
  target.
- The hourly table (about 31 columns, keyed by `study_id, hour, day`) is used
  for circadian features.
- `day` is stored as `YYYYMMDD`. The adapter converts it to a real date before
  doing any arithmetic, and drops 22 single-row `19691231` placeholder rows.

## College Experience Study

1. Download the archive from the Kaggle page (a Kaggle account is required).
2. Unzip it so the layout is:

```
data/raw/ces/
  Demographics/demographics.csv
  EMA/general_ema.csv
  EMA/covid_ema.csv
  Sensing/sensing.csv
  Sensing/steps.csv
  Readme.md
```

(`Raw Sensing/` is present in the archive and is not used.) CES reports PHQ-4;
the adapter recomputes the total from the four items and stores its negation so
that lower is worse in both cohorts.

## Checking that you have the right files

Run the audit on each cohort. It reads the data through the real adapter,
prints aggregate counts only, and includes a SHA-256 over the input files:

```bash
adaptivecare-data-audit --config configs/benchmark/forward_crosscheck.yaml
adaptivecare-data-audit --config configs/benchmark/forward_ces.yaml
```

The copies used to produce the reported results give:

| Cohort | `data_file_count` | `participants` | `candidate_landmarks` | `eligible_proxy_episodes` | `data_combined_sha256` |
|---|---|---|---|---|---|
| CrossCheck | 3 | 64 | 5,868 | 54 | `8585f7701a29529df059e5d12d666a9830df3a9fe4c6635947bf7cee9cba49c3` |
| CES | 267 | 210 | 32,967 | 127 | `a53972c86970b072a6d7ebec6219194c78f89d07dc34de01c57b8083cefff66e` |

If your counts match but the hash differs, the files differ byte for byte (for
example a re-exported CSV, or a different CES file set). Results may then
differ slightly, and the difference is worth reporting.

The digest covers every file under the cohort's folder, so keep extra files
out of `data/raw/crosscheck/` and `data/raw/ces/`. The CrossCheck count of 3
includes the data dictionary.

## Sensitive content

CrossCheck's EMA items include acute psychiatric symptom content
(`ema_HARM`, `ema_SEEING_THINGS`, `ema_VOICES`). The code uses the items as
numeric inputs and reports only aggregates. Keep it that way in your own
analyses.
