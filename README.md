# IPL Powerplay Prediction System

> Pre-match prediction of powerplay aggression and run-scoring — using historical ball-by-ball data, XGBoost regressors, and SHAP explanations served through a Flask web interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange)
![Flask](https://img.shields.io/badge/Flask-Web_App-lightgrey?logo=flask&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

---

## Overview

This system processes ball-by-ball data from 1,169 IPL matches (2008–2025), enriched with venue statistics from 10,984 additional T20 matches across 20 global competitions (BBL, PSL, CPL, T20I, and others). Given two teams and a venue, it predicts how aggressively each team is likely to bat in the powerplay (overs 1–6) — before the match starts.

The pipeline runs in five sequential stages: raw data ingestion → normalization → feature engineering → model training → web serving.

---

## Results

Validated using walk-forward cross-validation — trained strictly on past seasons, tested on the next unseen season.

| Metric | Value |
|---|---|
| MAE — Aggression % | **4.87 ± 0.55 pp** |
| MAE — Powerplay Runs | **11.59 ± 0.71** |
| Mean F1 (category label) | **0.304 ± 0.070** |
| Validation method | 5-fold walk-forward cross-validation |
| Primary training data | 1,169 IPL matches (2008–2025) |
| Venue enrichment data | 10,984 T20 matches across 20 competitions |

An aggression MAE of 4.87 pp means the model's prediction lands within ~5 percentage points of the actual value, on average, across five held-out seasons it had never seen during training.

---

## How It Works

Every prediction flows through a 5-stage pipeline:

```
Cricsheet ZIP Archives (IPL + 20 T20 competitions)
        │
        ▼
  parse_cricsheet.py ── Extracts match JSONs → normalizes team/venue names
        │                → outputs balls.csv, matches.csv, powerplay_summary.csv
        ▼
  feature_builder.py ── Reads processed CSVs → engineers 17 features per innings
        │                → outputs training_matrix.csv
        │                Prints null-rate report (expected for new franchises)
        ▼
  train.py ──────────── Trains two XGBoost regressors:
        │                 · Aggression % predictor  → regressor_aggression.pkl
        │                 · Powerplay runs predictor → regressor_runs.pkl
        │                Saves models + metadata to models/
        ▼
  evaluate.py ────────── Walk-forward cross-validation across 5 folds
        │                (train on past seasons → test on next unseen season)
        │                Reports MAE per fold + mean ± std
        ▼
  app.py (Flask) ─────── Loads saved models → serves predictions via web UI
                          Runs SHAP explainer on demand per prediction
```

---

## Feature Engineering

`feature_builder.py` builds a 17-feature training matrix — one row per IPL innings from 2016 onwards. Every feature is computable before the match starts. No in-match data is used.

| Feature | Source |
|---|---|
| Team powerplay aggression — last season | IPL balls.csv |
| Team powerplay aggression — two seasons ago | IPL balls.csv |
| Weighted 2-season average | Derived |
| Year-on-year aggression trend | Derived |
| Team average powerplay runs — last season | IPL balls.csv |
| Team wicket loss rate in powerplay | IPL balls.csv |
| Opposition bowling economy in powerplay | IPL balls.csv |
| Opposition wicket-taking rate in powerplay | IPL balls.csv |
| Opposition economy trend | Derived |
| Head-to-head powerplay history (these two teams) | IPL balls.csv |
| Venue average powerplay aggression | T20 + IPL combined |
| Venue average powerplay runs | T20 + IPL combined |
| Venue innings count (data reliability weight) | T20 + IPL combined |
| Innings number (1st or 2nd) | matches.csv |
| Day-Night flag | matches.csv |
| Season era index | Derived |
| Playoff flag | matches.csv |

---

## SHAP Explanations

After every prediction, clicking **"Explain this prediction"** runs a SHAP analysis that shows exactly which features pushed the score up or down — not just the final number. For example: *"Wankhede Stadium's high-scoring history added +1.5%, but Mumbai Indians' low aggression last season pulled it down by −0.9%."* This turns a black-box number into an interpretable breakdown per feature.

---

## Data Sources

**IPL data** — 1,169 matches, 2008–2025, ball-by-ball JSON format from [Cricsheet.org](https://cricsheet.org).

**T20 enrichment data** — 10,984 matches across BBL, PSL, CPL, T20I, and 16 other competitions. Used exclusively for venue-level baseline statistics (average powerplay aggression and scoring by ground).

Both sources ship as ZIP archives containing one JSON file per match. Key normalization tasks handled by `parse_cricsheet.py`:

- **Franchise rename mapping** — IPL teams that have changed names across seasons (e.g., Delhi Daredevils → Delhi Capitals) are remapped to current franchise names across all historical records.
- **Venue deduplication** — the same ground often appears under multiple spellings and abbreviations across competitions. A venue map consolidates these into canonical names.
- **Competition tagging** — each innings is tagged with its source competition so IPL and non-IPL records can be filtered independently downstream.

---

## Project Structure

```
ipl_powerplay_ml/
│
├── data/
│   ├── ipl/                        IPL zip from Cricsheet
│   ├── t20s/                       Other T20 zips (BBL, PSL, CPL, etc.)
│   ├── people.csv                  Player ID registry
│   └── processed/                  Generated CSVs (auto-created by pipeline)
│
├── src/
│   ├── parse_cricsheet.py          ZIP extraction, JSON parsing, CSV output
│   ├── team_names.py               Franchise rename + venue name maps
│   ├── feature_builder.py          17-feature training matrix builder
│   ├── train.py                    Model training + serialization
│   ├── evaluate.py                 Walk-forward cross-validation + metrics
│   ├── predict.py                  Pre-match inference
│   └── explain_model.py            SHAP feature attribution
│
├── models/                         Saved model files (auto-created by train.py)
│   ├── regressor_aggression.pkl
│   ├── regressor_runs.pkl
│   └── meta.json
│
├── templates/
│   └── index.html                  Web interface
│
├── app.py                          Flask web server
└── requirements.txt
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ipl-powerplay-ml.git
cd ipl-powerplay-ml
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy xgboost==1.7.6 scikit-learn shap joblib matplotlib seaborn flask jupyter
```

> **Use `xgboost==1.7.6` specifically.** Later versions serialize the model's base score in a format that SHAP cannot parse.

### 4. Place data files

```
data/
├── ipl/
│   └── ipl.zip           ← from cricsheet.org
├── t20s/
│   ├── bbl_json.zip
│   ├── cpl_json.zip
│   └── ...               ← all other T20 zips
├── people.csv
└── names.csv
```

---

## Running the Pipeline

Run these steps in order from the project root. Each step depends on the previous one's output.

### Step 1 — Parse IPL data

```bash
python src/parse_cricsheet.py data/ipl --competition IPL --out data/processed/ipl
```

Outputs: `balls.csv`, `matches.csv`, `powerplay_summary.csv`

### Step 2 — Parse T20 enrichment data

```bash
python src/parse_cricsheet.py data/t20s --out data/processed/t20s
```

Outputs venue baseline CSVs used in feature engineering.

### Step 3 — Build training matrix

```bash
python src/feature_builder.py --out data/processed/training_matrix.csv
```

Engineers all 17 features for every IPL innings from 2016 onwards. Prints a null-rate report — some nulls are expected for new franchises with no prior history.

### Step 4 — Train models

```bash
python src/train.py
```

Trains two XGBoost regressors and saves them to `models/`. Prints feature importances on completion — `season_index` will typically rank highest.

### Step 5 — Evaluate (optional but recommended)

```bash
python src/evaluate.py
```

Runs 5-fold walk-forward cross-validation. Reports out-of-sample MAE per fold and mean ± std.

### Step 6 — Start the web app

```bash
python app.py
```

Open: **http://localhost:5000**

---

## Evaluation Results

| Fold | Train Seasons | Test Season | MAE Aggression | MAE Runs | F1 |
|---|---|---|---|---|---|
| 1 | 2016–2019 | 2021 | 5.55 pp | 12.48 | 0.201 |
| 2 | 2016–2021 | 2022 | 4.50 pp | 11.38 | 0.266 |
| 3 | 2016–2022 | 2023 | 4.15 pp | 10.62 | 0.366 |
| 4 | 2016–2023 | 2024 | 5.04 pp | 12.06 | 0.359 |
| 5 | 2016–2024 | 2025 | 5.11 pp | 11.39 | 0.326 |
| **Mean ± Std** | | | **4.87 ± 0.55 pp** | **11.59 ± 0.71** | **0.304 ± 0.070** |

The runs MAE is higher than aggression MAE because powerplay scoring is more sensitive to pitch conditions and dew — factors not available as pre-match inputs.

---

## Web App

### Predict

Select Team A (batting first), Team B, and a venue. Home ground chips appear automatically. Choose season and match flags, then submit. SHAP feature attribution is available per prediction — shows exactly which inputs pushed the score up or down and by how much.

### Leaderboard

All IPL teams ranked by historical powerplay aggression, filterable by season.

### Venues

All IPL venues ranked by average powerplay aggression, filterable by usage frequency (Regular / Occasional / Rare).

---

## Updating with New Data

When new IPL matches are played, re-run Steps 1 → 3 → 4. Steps 2 and 5 only need re-running if you add new T20 zip files or want updated evaluation metrics.

---

## Known Limitations

- **No pitch or weather data** — toss result, pitch conditions, and dew are not modelled. These are the primary unexplained variance source in runs prediction.
- **New franchises** — teams with no prior IPL history produce partial feature vectors. The model handles these via XGBoost's native NaN routing.
- **Missing season** — IPL 2020 is absent from Cricsheet's standard ZIP. The pipeline handles the gap automatically.
- **Venue threshold sensitivity** — venue baselines rely on T20 enrichment data; rarely-used grounds have lower sample counts and correspondingly wider prediction intervals (reflected in the `venue_innings_count` feature).

---

## Stack

| Component | Library | Role |
|---|---|---|
| Data Ingestion | Python stdlib (zipfile, json) | JSON extraction from ZIP archives |
| Data Processing | Pandas, NumPy | CSV normalization, feature engineering |
| Modelling | XGBoost 1.7.6 | Aggression % and runs regression |
| Explainability | SHAP | Per-prediction feature attribution |
| Validation | scikit-learn | Walk-forward cross-validation |
| Web Interface | Flask | Pre-match prediction UI |
| Serialization | joblib | Model persistence |
| Visualization | Matplotlib, Seaborn | Evaluation graphs |

---

## Future Improvements

- Incorporate **toss result and pitch report** as late-binding features at prediction time
- Add a **confidence interval display** in the web UI based on historical variance per team/venue
- Build a **live update hook** — auto-fetch and parse new Cricsheet data after each IPL match
- Extend predictions to **full innings total** using powerplay trajectory as an early signal
- Add **player-level features** — key powerplay batters' recent form and strike rates
- Replace XGBoost with a **sequence model (LSTM / Transformer)** on ball-by-ball history for richer temporal patterns

---

## Dataset

**Cricsheet.org**
- IPL: 1,169 matches, 2008–2025, ball-by-ball JSON format
- T20 enrichment: 10,984 matches across BBL, PSL, CPL, T20I, and 16 other competitions
- Source: [https://cricsheet.org/downloads/](https://cricsheet.org/downloads/)

---

## References

- Chen & Guestrin (2016) — XGBoost: A Scalable Tree Boosting System. KDD 2016.
- Lundberg & Lee (2017) — A Unified Approach to Interpreting Model Predictions (SHAP). NeurIPS 2017.
- Cricsheet.org — Ball-by-ball match data. [https://cricsheet.org](https://cricsheet.org)

---

*Built as part of Project 2 — 5th Semester, B.E. Information Technology, 2025–2026*
