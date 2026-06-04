"""
train.py
========
Trains all three XGBoost models on the full training matrix and saves
them to the models/ folder.

Models trained:
    Model A  regressor_aggression.pkl   target: pp_aggression (%)
    Model B  regressor_runs.pkl         target: pp_runs (integer)
    Model C  classifier_label.pkl       target: H / M / L label
                                        (derived from pp_aggression percentiles)

The H/M/L thresholds are computed from the training data and saved
alongside the models so the classifier and evaluate/predict scripts
use identical boundaries.

Usage
-----
    python src/train.py

    # Use a different training matrix (e.g. after re-parsing)
    python src/train.py --matrix data/processed/training_matrix.csv

    # Save models to a different folder
    python src/train.py --models-dir models/v2
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_MATRIX    = Path("data/processed/training_matrix.csv")
DEFAULT_MODELS    = Path("models")

# ---------------------------------------------------------------------------
# Feature columns — must match feature_builder.py exactly
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "bat_pp_aggression_prev1",
    "bat_pp_aggression_prev2",
    "bat_pp_aggression_2s_avg",
    "bat_pp_aggression_trend",
    "bat_pp_runs_prev1",
    "bat_pp_wicket_rate_prev1",
    "bowl_pp_economy_prev1",
    "bowl_pp_wicket_rate_prev1",
    "bowl_pp_economy_trend",
    "h2h_pp_aggression",
    "venue_pp_aggression",
    "venue_pp_runs",
    "venue_innings_count",
    "innings_num",
    "is_day_night",
    "season_index",
    "is_playoff",
]

TARGET_AGGRESSION = "pp_aggression"
TARGET_RUNS       = "pp_runs"

# ---------------------------------------------------------------------------
# XGBoost hyperparameters
# Conservative settings for ~1,200 rows — avoids overfitting
# ---------------------------------------------------------------------------

REGRESSOR_PARAMS = {
    "n_estimators":       300,
    "max_depth":          4,
    "learning_rate":      0.05,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "min_child_weight":   5,
    "reg_alpha":          0.1,    # L1
    "reg_lambda":         1.0,    # L2
    "base_score":         0.5,    # 
    "random_state":       42,
    "n_jobs":             -1,
    "tree_method":        "hist", # fast on CPU
}

CLASSIFIER_PARAMS = {
    **REGRESSOR_PARAMS,
    "objective":          "multi:softprob",
    "eval_metric":        "mlogloss",
    "num_class":          3,
}


# ---------------------------------------------------------------------------
# Label derivation
# ---------------------------------------------------------------------------

def make_labels(
    aggression: pd.Series,
) -> "tuple[np.ndarray, float, float]":
    """
    Derive H / M / L labels from aggression % using percentile thresholds.

    Thresholds are computed from the data (not hardcoded) so the three
    classes are always roughly balanced regardless of the season range used.

    Returns:
        labels      : integer array  0=Low, 1=Medium, 2=High
        low_thresh  : aggression % below this -> Low
        high_thresh : aggression % above this -> High
    """
    low_thresh  = float(np.nanpercentile(aggression, 33))
    high_thresh = float(np.nanpercentile(aggression, 67))

    labels = np.where(
        aggression > high_thresh, 2,          # High
        np.where(aggression < low_thresh, 0,  # Low
        1)                                     # Medium
    )
    return labels, low_thresh, high_thresh


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    matrix_path: str = str(DEFAULT_MATRIX),
    models_dir:  str = str(DEFAULT_MODELS),
    verbose:     bool = True,
) -> dict:
    """
    Load the training matrix, fit all three models, save to models_dir.

    Args:
        matrix_path : path to training_matrix.csv
        models_dir  : folder to save .pkl files and thresholds.json
        verbose     : print progress and metrics

    Returns:
        dict with trained models and threshold values
    """
    # ---- Load ----------------------------------------------------------- #
    matrix_path = Path(matrix_path)
    models_dir  = Path(models_dir)

    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Training matrix not found: {matrix_path}\n"
            f"Run first:  python src/feature_builder.py"
        )

    if verbose:
        print(f"Loading training matrix from: {matrix_path}")

    df = pd.read_csv(matrix_path)

    if verbose:
        print(f"  Rows   : {len(df):,}")
        print(f"  Cols   : {df.shape[1]}")

    # ---- Validate feature columns --------------------------------------- #
    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"Training matrix is missing expected feature columns:\n"
            f"  {missing_features}\n"
            f"Re-run feature_builder.py to regenerate the matrix."
        )

    missing_targets = [
        c for c in [TARGET_AGGRESSION, TARGET_RUNS] if c not in df.columns
    ]
    if missing_targets:
        raise ValueError(f"Training matrix missing target columns: {missing_targets}")

    # ---- Prepare X and y ------------------------------------------------ #
    # Drop rows where ALL features are NaN (should not happen, but safety)
    df = df.dropna(subset=[TARGET_AGGRESSION, TARGET_RUNS])

    X = df[FEATURE_COLS].copy()
    y_aggression = df[TARGET_AGGRESSION].astype(float)
    y_runs       = df[TARGET_RUNS].astype(float)

    # XGBoost handles NaN natively via its missing-value split mechanism.
    # No imputation needed — leave NaN as-is.

    if verbose:
        print(f"\nFeature matrix shape : {X.shape}")
        print(f"NaN cells            : {X.isna().sum().sum():,}  "
              f"({X.isna().mean().mean():.1%} of all cells)")

    # ---- H/M/L labels --------------------------------------------------- #
    labels, low_thresh, high_thresh = make_labels(y_aggression)
    label_map = {0: "Low", 1: "Medium", 2: "High"}

    if verbose:
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\nH/M/L thresholds:")
        print(f"  Low    < {low_thresh:.2f}%   ({counts[0]:,} rows)")
        print(f"  Medium   {low_thresh:.2f}% – {high_thresh:.2f}%   ({counts[1]:,} rows)")
        print(f"  High   > {high_thresh:.2f}%   ({counts[2]:,} rows)")

    # ---- Train Model A — aggression regressor -------------------------- #
    if verbose:
        print("\n[Model A] Training aggression regressor...")

    model_a = XGBRegressor(**REGRESSOR_PARAMS)
    model_a.fit(X, y_aggression)

    train_mae_a = float(np.mean(np.abs(model_a.predict(X) - y_aggression)))
    if verbose:
        print(f"  Train MAE (aggression %) : {train_mae_a:.4f}")
        print("  Note: train MAE is optimistic — see evaluate.py for honest scores")

    # ---- Train Model B — runs regressor -------------------------------- #
    if verbose:
        print("\n[Model B] Training runs regressor...")

    model_b = XGBRegressor(**REGRESSOR_PARAMS)
    model_b.fit(X, y_runs)

    train_mae_b = float(np.mean(np.abs(model_b.predict(X) - y_runs)))
    if verbose:
        print(f"  Train MAE (runs)         : {train_mae_b:.4f}")

    # ---- Train Model C — H/M/L classifier ------------------------------ #
    if verbose:
        print("\n[Model C] Training H/M/L classifier...")

    model_c = XGBClassifier(**CLASSIFIER_PARAMS)
    model_c.fit(X, labels)

    train_preds_c = model_c.predict(X)
    train_acc_c   = float(np.mean(train_preds_c == labels))
    if verbose:
        print(f"  Train accuracy (H/M/L)   : {train_acc_c:.4f}")

    # ---- Feature importance summary ------------------------------------ #
    if verbose:
        print("\nFeature importances (Model A — aggression regressor):")
        importances = model_a.feature_importances_
        pairs = sorted(
            zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True
        )
        for feat, imp in pairs:
            bar = "█" * int(imp * 200)
            print(f"  {feat:<38} {imp:.4f}  {bar}")

    # ---- Save models ---------------------------------------------------- #
    models_dir.mkdir(parents=True, exist_ok=True)

    path_a = models_dir / "regressor_aggression.pkl"
    path_b = models_dir / "regressor_runs.pkl"
    path_c = models_dir / "classifier_label.pkl"

    joblib.dump(model_a, path_a)
    joblib.dump(model_b, path_b)
    joblib.dump(model_c, path_c)

    # Save thresholds and metadata alongside models
    meta = {
        "low_thresh":     low_thresh,
        "high_thresh":    high_thresh,
        "label_map":      label_map,
        "feature_cols":   FEATURE_COLS,
        "train_rows":     len(df),
        "train_mae_aggression": train_mae_a,
        "train_mae_runs":       train_mae_b,
        "train_acc_label":      train_acc_c,
    }
    meta_path = models_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"\nSaved to: {models_dir}/")
        print(f"  regressor_aggression.pkl")
        print(f"  regressor_runs.pkl")
        print(f"  classifier_label.pkl")
        print(f"  meta.json")
        print(f"\nDone. Run evaluate.py next for honest walk-forward metrics.")

    return {
        "model_aggression": model_a,
        "model_runs":       model_b,
        "model_label":      model_c,
        "low_thresh":       low_thresh,
        "high_thresh":      high_thresh,
        "meta":             meta,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Train all three XGBoost models and save to models/."
    )
    ap.add_argument(
        "--matrix",
        default=str(DEFAULT_MATRIX),
        help=f"Path to training matrix CSV (default: {DEFAULT_MATRIX})",
    )
    ap.add_argument(
        "--models-dir",
        default=str(DEFAULT_MODELS),
        help=f"Folder to save model .pkl files (default: {DEFAULT_MODELS})",
    )
    args = ap.parse_args()

    train(
        matrix_path = args.matrix,
        models_dir  = args.models_dir,
        verbose     = True,
    )