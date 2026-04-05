# Predictive Collision Severity System: Using Machine Learning to Forecast Time-to-Collision (TTC) and Impact Force from the FIUS Data

This repository implements a machine-learning pipeline for proactive collision detection from raw FIUS ultrasonic sensor data. Instead of relying only on the firmware-reported distance value, the project analyzes the raw ADC echo waveform, estimates object motion with a Kalman filter, labels approaching motion automatically, trains three downstream models, and simulates real-time inference for collision timing and severity estimation.

The end-to-end orchestrator is [`run_pipeline.sh`](run_pipeline.sh).

## Overview

The project is organized around six practical stages:

1. Ingest headerless raw CSV recordings from `data/raw/`.
2. Extract per-frame physical, spectral, and temporal features into `data/processed/features.csv`.
3. Generate motion labels and quadratic time-to-collision targets in `data/processed/final_labeled_data.csv`.
4. Train motion, TTC, and material models and serialize them to `models/`.
5. Replay raw sessions through the trained models with `src/models/predictor.py`.
6. Export evaluation reports and figures into `reports/`.

At code level, the repository currently contains:

- 4 raw sensor recordings
- 7,000 processed frames
- 1,343 `towards` frames in the checked-in labeled dataset
- 1,337 finite TTC labels in the checked-in labeled dataset
- 3 serialized production model artifacts in `models/`

## Project Status and Evidence Sources

This repository includes two different sources of truth:

- The code and checked-in artifacts in `data/processed/`, `models/`, and `reports/`
- The academic write-up in [`TOW-1 Gud Shi Ban Pat FinalReport.pdf`](TOW-1%20Gud%20Shi%20Ban%20Pat%20FinalReport.pdf)

They do not currently tell exactly the same performance story, so the distinction should be explicit.

| Source | What it shows |
| --- | --- |
| Checked-in `reports/detailed_algorithm_performance.csv` | Naive baseline F1 = `0.701`, best ML OCC result = `0.512` (Isolation Forest), OCSVM F1 = `0.385` |
| Checked-in `reports/confusion_matrix.csv` | Motion confusion matrix for the selected trained detector: TP `267`, FN `97`, FP `412`, TN `724` |
| Checked-in processed data | `features.csv` shape = `7000 x 21`, `final_labeled_data.csv` shape = `7000 x 24` |
| [`TOW-1 Gud Shi Ban Pat FinalReport.pdf`](TOW-1%20Gud%20Shi%20Ban%20Pat%20FinalReport.pdf) | Reports stronger experimental results, including OCSVM F1 around `0.92`, SVR MAE around `0.12 s`, material accuracy around `0.91`, and average inference latency around `0.85 ms` |

If you are using this repository for coursework, reporting, or further development, treat the checked-in reports and the paper as separate evidence unless you rerun the full pipeline and regenerate the metrics yourself.

## Pipeline Architecture

```text
                               +----------------------+
                               |   data/raw/*.csv     |
                               |   Raw FIUS sessions  |
                               +----------+-----------+
                                          |
                                          v
                         +----------------+----------------+
                         | src/data/feature_extractor.py   |
                         | DC centering, echo detection,   |
                         | FFT, Kalman-derived features,   |
                         | temporal window features        |
                         +----------------+----------------+
                                          |
                                          v
                          +---------------+---------------+
                          | data/processed/features.csv   |
                          +---------------+---------------+
                                          |
                                          v
                         +----------------+----------------+
                         | src/data/label_generator.py     |
                         | motion labels, distance, TTC    |
                         +----------------+----------------+
                                          |
                                          v
                   +----------------------+----------------------+
                   | data/processed/final_labeled_data.csv       |
                   +----------------------+----------------------+
                                          |
                                          v
                         +----------------+----------------+
                         | src/models/model_trainer.py     |
                         | OCC sweep, TTC regression,      |
                         | material classification         |
                         +-----------+-----------+---------+
                                     |           |
                      writes models  |           | writes reports
                                     v           v
                    +-------------------+   +----------------------+
                    |   models/*.joblib |   |    reports/*.csv     |
                    +---------+---------+   +----------+-----------+
                              |                        |
                              |                        |
          +-------------------+                        +------------------+
          |                                                               |
          v                                                               v
+---------+----------------+                           +------------------+------------------+
| src/models/predictor.py  |                           | src/visualization/*.py              |
| frame-by-frame inference |                           | plots from raw data, processed      |
| motion, TTC, material,   |                           | datasets, and training reports      |
| impact force             |                           +------------------+------------------+
+------------+-------------+                                              |
             |                                                            v
             v                                           +----------------+------------------+
+------------+-------------+                             |     reports/figures/*.png         |
| Console inference report |                             +-----------------------------------+
+--------------------------+
```

## How the System Works

### 1. Raw data ingestion

Raw recordings are read from `data/raw/`. The pipeline expects headerless CSV files where:

- Columns `0-9` contain session metadata
- Column `10` contains the firmware-reported distance measurement
- Columns `11-15` contain additional metadata
- Column `16` contains the frame timestamp in milliseconds
- Columns `17+` contain raw ADC samples

The extractor assumes the ADC payload starts at column `17`, which is defined centrally as `ADC_DATA_START_INDEX` in `src/data/processing.py`.

### 2. Signal preprocessing and echo detection

`src/data/feature_extractor.py` and `src/models/predictor.py` share the same signal-processing rules:

- Frame-wise DC centering of the ADC waveform
- Adaptive first-echo detection via `find_first_peak_index(...)`
- Shared blanking and threshold constants from `src/data/processing.py`
- Dynamic blanking near the transducer blind spot using the Kalman-estimated distance
- Echo shape extraction via amplitude and FWHM (`echo_width`)

Important constants:

- Sampling rate: `1,953,125 Hz`
- Speed of sound: `343.0 m/s`
- Static blanking start: `ECHO_MIN_INDEX = 20`
- Echo threshold: `ECHO_THRESHOLD_MULTIPLIER = 4`

### 3. Kinematic estimation

`src/data/kalman.py` implements a 1D constant-acceleration Kalman filter over:

- distance
- velocity
- acceleration

The filter recomputes its transition matrix from the actual frame-to-frame timestamp delta, which keeps training-time and inference-time kinematics aligned even when sampling jitter is present.

### 4. Feature engineering

`src/data/feature_extractor.py` generates a feature table with:

- Base per-frame features:
  - `echo_index`
  - `echo_amplitude`
  - `echo_width`
  - `Peak Frequency`
  - `Spectral Centroid`
- Temporal features at windows `[5, 10, 25, 50]`:
  - `Trend_w`
  - `Mean_w`
  - `Centroid_Trend_w`

This produces the 17-dimensional feature vector used by the motion classifier:

- 5 base features
- 12 rolling-window features

The checked-in `data/processed/features.csv` has 21 columns total because it also stores `label`, `timestamp`, `velocity`, and `acceleration`.

### 5. Label generation

`src/data/label_generator.py` creates the supervised training targets by:

- Thresholding Kalman velocity at `-0.05 m/s`
- Applying a centered majority-vote smoothing window of size `3`
- Eroding `2` frames from the boundaries of each `towards` segment
- Recomputing distance from `echo_index`
- Solving a quadratic TTC equation for approaching frames

The final output is `data/processed/final_labeled_data.csv`, which currently has 24 columns including:

- `session_id`
- all extracted features
- `calc_dist_m`
- `ttc`
- final `label`

### 6. Model training

`src/models/model_trainer.py` trains three model families:

#### Motion detection

One-class classification is trained only on `towards` frames and benchmarked across:

- One-Class SVM
- Isolation Forest
- Local Outlier Factor with `novelty=True`
- Autoencoder-based anomaly detector

Validation is session-aware through `GroupKFold` and `GroupShuffleSplit`, using `session_id` as the grouping key.

#### TTC regression

For frames labeled `towards` with valid TTC targets, the trainer compares:

- Linear Regression
- SVR with RBF kernel

#### Material classification

A `RandomForestClassifier` predicts the object/session label from:

- `echo_amplitude`
- `echo_width`
- `Spectral Centroid`
- `Peak Frequency`

Serialized artifacts:

- `models/motion_detection_model.joblib`
- `models/ttc_prediction_model.joblib`
- `models/material_classifier_model.joblib`

### 7. Real-time inference simulation

`src/models/predictor.py` replays one raw recording frame by frame. Once its temporal buffers are warm, it:

1. Rebuilds the same motion feature vector used during training.
2. Predicts whether the frame represents `towards` motion.
3. If `towards`, predicts TTC.
4. Predicts material/session class.
5. Estimates impact force using `F = m * |a|`.
6. Measures per-frame latency with `time.perf_counter()`.

Run it manually with:

```bash
./venv/bin/python src/models/predictor.py data/raw/signal_2000_people.csv
```

### 8. Visualization and reporting

The repository includes plotting scripts for:

- label distribution
- raw echo detection
- classifier comparison
- confusion matrix
- TTC prediction scatter
- echo profiles
- velocity timeline
- feature correlation
- distance-over-time plots for each raw file

Outputs are saved under `reports/figures/`.

## Quick Start

### Prerequisites

- Python `3.10+` is recommended
- macOS or Linux shell environment
- `pip` and `venv`

The checked-in virtual environment in this repository uses Python `3.14.2`, but creating a fresh environment is still the cleaner setup path.

### Recommended setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH="$(pwd)"
```

### Run the full pipeline

```bash
bash run_pipeline.sh
```

What `run_pipeline.sh` does:

1. Deletes old model artifacts, report CSVs, and figures
2. Creates or reuses `venv/`
3. Installs dependencies from `requirements.txt`
4. Runs feature extraction
5. Runs label generation
6. Runs model training
7. Replays prediction on every raw CSV file
8. Generates all figures
9. Writes a combined console log to `processing.log`

Important: the script starts with a cleanup step and removes prior generated outputs from `models/` and `reports/`.

## Run Components Individually

If you do not want to execute the full orchestrator, the main entrypoints are:

```bash
./venv/bin/python src/data/feature_extractor.py
./venv/bin/python src/data/label_generator.py
./venv/bin/python src/models/model_trainer.py
./venv/bin/python src/models/predictor.py data/raw/signal_2000_people.csv
./venv/bin/python src/visualization/plot_label_distribution.py
./venv/bin/python src/visualization/plot_raw_echo.py
./venv/bin/python src/visualization/plot_classifier_comparison.py
./venv/bin/python src/visualization/plot_confusion_matrix.py
./venv/bin/python src/visualization/plot_ttc_prediction.py
./venv/bin/python src/visualization/plot_echo_profiles.py
./venv/bin/python src/visualization/plot_velocity_timeline.py
./venv/bin/python src/visualization/plot_feature_correlation.py
./venv/bin/python src/visualization/plot_distance_over_time.py data/raw/signal_2000_people.csv
```

## Dataset and Naming Convention

All raw files must follow:

```text
signal_{id}_{object_name}.csv
```

Examples currently in the repository:

- `signal_1500_metal_plate.csv`
- `signal_1500_people_with_keeping_distance.csv`
- `signal_2000_cardboard.csv`
- `signal_2000_people.csv`

Current raw-session sizes:

| File | Frames |
| --- | ---: |
| `signal_1500_metal_plate.csv` | 1500 |
| `signal_1500_people_with_keeping_distance.csv` | 1500 |
| `signal_2000_cardboard.csv` | 2000 |
| `signal_2000_people.csv` | 2000 |

The object label is derived from the filename and propagated through the pipeline. Additional details are documented in [`DATA_CONVENTION.md`](DATA_CONVENTION.md).

## Generated Artifacts

| Path | Produced by | Purpose |
| --- | --- | --- |
| `data/processed/features.csv` | `feature_extractor.py` | Extracted sensor and temporal features |
| `data/processed/final_labeled_data.csv` | `label_generator.py` | Training dataset with labels and TTC targets |
| `models/motion_detection_model.joblib` | `model_trainer.py` | Selected motion detector |
| `models/ttc_prediction_model.joblib` | `model_trainer.py` | Selected TTC regressor |
| `models/material_classifier_model.joblib` | `model_trainer.py` | Material/session classifier |
| `reports/detailed_algorithm_performance.csv` | `model_trainer.py` | Motion-classifier benchmark results |
| `reports/confusion_matrix.csv` | `model_trainer.py` | Confusion matrix for selected motion model |
| `reports/figures/*.png` | visualization scripts | Generated plots for analysis and reporting |
| `processing.log` | `run_pipeline.sh` | Full orchestrator log |

## Repository Layout

```text
.
├── data/
│   ├── raw/                  # headerless FIUS recordings
│   └── processed/            # features.csv and final_labeled_data.csv
├── docs/                     # supplemental FIUS documentation
├── models/                   # serialized .joblib artifacts
├── reports/
│   ├── *.csv                 # evaluation tables
│   └── figures/              # generated plots
├── src/
│   ├── data/                 # preprocessing, Kalman filter, labeling
│   ├── models/               # training, inference, detectors
│   └── visualization/        # reporting plots
├── DATA_CONVENTION.md
├── TOW-1 Gud Shi Ban Pat FinalReport.pdf
├── requirements.txt
└── run_pipeline.sh
```

## Key Implementation Constraints

These details matter if you extend the project:

- Feature extraction and predictor logic share constants from `src/data/processing.py`; changing them in one place without the other will create training/inference drift.
- The motion classifier trains on `towards` frames only because it is implemented as one-class classification.
- Session grouping is derived from filenames, not from a separate metadata table.
- The plotting scripts assume the processed datasets and reports already exist.
- The repository currently contains only four sessions, so generalization claims should be treated cautiously.

## Known Caveats

### 1. Paper metrics vs repository metrics

The checked-in `reports/` files do not currently match the stronger values described in `TOW-1 Gud Shi Ban Pat FinalReport.pdf`. Do not present them as if they were the same experiment unless you regenerate the artifacts and confirm the numbers yourself.

### 2. Force estimation label mismatch

`src/models/predictor.py` maps predicted material labels to masses with:

- `human -> 70.0 kg`
- `metal_plate -> 2.0 kg`
- `cardboard -> 0.5 kg`

However, the current trained material classifier exposes these classes:

- `cardboard`
- `metal_plate`
- `people`
- `people_with_keeping_distance`

That means labels such as `people` do not match the hardcoded `human` key and therefore fall back to the default mass of `1.0 kg`. If force estimation matters, update `MASS_ESTIMATES` to match your actual label vocabulary.

### 3. Headerless raw format

The entire pipeline relies on hard-coded column positions. If the FIUS export format changes, you must update the relevant indices before rerunning training.

## Supporting Documents

- [`TOW-1 Gud Shi Ban Pat FinalReport.pdf`](TOW-1%20Gud%20Shi%20Ban%20Pat%20FinalReport.pdf): academic write-up, methodology, reported results, limitations, and references
- [`DATA_CONVENTION.md`](DATA_CONVENTION.md): required raw-file naming scheme
- [`docs/introduction to fius.pdf`](docs/introduction%20to%20fius.pdf): additional FIUS context
- [`docs/header_structure.xlsx`](docs/header_structure.xlsx): raw-header reference material

## Recommended Next Steps

If you want to continue developing this project, the highest-value follow-ups are:

1. Regenerate all reports from scratch and reconcile them with the numbers in `TOW-1 Gud Shi Ban Pat FinalReport.pdf`.
2. Fix the material-to-mass label mismatch in `predictor.py`.
3. Add a small validation script that checks raw CSV schema, filename convention, and artifact freshness before training.
4. Expand the dataset beyond four sessions before making stronger generalization claims.

## Authors

| Member | Name                      | GitHub ID                                                |
| ------ | ------------------------- | -------------------------------------------------------- |
| 1      | `Aditya Praveen Shidhaye` | [@axiomsbane](https://github.com/axiomsbane)             |
| 2      | `Priyanshu Bandyopadhyay` | [@prnshubn](https://github.com/prnshubn)                 |
| 3      | `Pradeep Buddhiram Patwa` | [@pradeeppatwa](https://github.com/pradeeppatwa)         |
| 4      | `Akshay Ganesh Gudekar`   | [@Akshay-Gudekar](https://github.com/Akshay-Gudekar)     |
