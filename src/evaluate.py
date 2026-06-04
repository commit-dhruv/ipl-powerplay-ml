"""
evaluate.py
===========
Walk-forward cross-validation for all three IPL powerplay ML models.

Each fold trains on all seasons up to year N and tests on season N+1.
This mirrors real deployment — the model never sees future data.

Folds (default):
    Fold 1 : train 2016-2019  test 2021
    Fold 2 : train 2016-2021  test 2022
    Fold 3 : train 2016-2022  test 2023
    Fold 4 : train 2016-2023  test 2024
    Fold 5 : train 2016-2024  test 2025

Note: 2020 is missing from the dataset (IPL 2020 not present in this zip).
The walk-forward logic handles gaps automatically — it trains on whatever
seasons exist before the test season.

Metrics reported:
    Model A (aggression regressor) : MAE, RMSE
    Model B (runs regressor)       : MAE, RMSE
    Model C (H/M/L classifier)     : Accuracy, Macro F1

Usage
-----
    python src/evaluate.py

    # Custom matrix path
    python src/evaluate.py --matrix data/processed/training_matrix.csv

    # Save results to CSV
    python src/evaluate.py --out data/processed/eval_results.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from xgboost import XGBClassifier, XGBRegressor

from train import (
    CLASSIFIER_PARAMS,
    FEATURE_COLS,
    REGRESSOR_PARAMS,
    TARGET_AGGRESSION,
    TARGET_RUNS,
    make_labels,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_MATRIX = Path("data/processed/training_matrix.csv")

# ---------------------------------------------------------------------------
# Walk-forward fold definitions
# ---------------------------------------------------------------------------

# Each tuple: (last train season, test season)
# Season 2020 is absent from dataset — skipped automatically
FOLDS = [
    (2019, 2021),
    (2021, 2022),
    (2022, 2023),
    (2023, 2024),
    (2024, 2025),
]


# ---------------------------------------------------------------------------
# Single fold evaluation
# ---------------------------------------------------------------------------

def _run_fold(
    df:          pd.DataFrame,
    train_up_to: int,
    test_season: int,
    low_thresh:  float,
    high_thresh: float,
) -> "dict | None":
    """
    Train on seasons <= train_up_to, test on test_season.

    Args:
        df          : full training matrix
        train_up_to : last season included in training window
        test_season : season held out for testing
        low_thresh  : aggression % below which -> Low label
        high_thresh : aggression % above which -> High label

    Returns:
        dict of metrics for this fold, or None if insufficient data.
    """
    train_mask = (df["season"] >= 2016) & (df["season"] <= train_up_to)
    test_mask  = df["season"] == test_season

    train_df = df[train_mask].copy()
    test_df  = df[test_mask].copy()

    if len(train_df) < 50:
        print(f"  [SKIP] Fold train<='{train_up_to}': only {len(train_df)} train rows")
        return None
    if len(test_df) == 0:
        print(f"  [SKIP] Fold test='{test_season}': no test rows (season not in data)")
        return None

    X_train = train_df[FEATURE_COLS]
    X_test  = test_df[FEATURE_COLS]

    y_train_agg  = train_df[TARGET_AGGRESSION].astype(float)
    y_test_agg   = test_df[TARGET_AGGRESSION].astype(float)
    y_train_runs = train_df[TARGET_RUNS].astype(float)
    y_test_runs  = test_df[TARGET_RUNS].astype(float)

    # Derive labels using the GLOBAL thresholds (computed from full training set)
    # This ensures consistent class boundaries across all folds
    y_train_lbl = np.where(
        y_train_agg > high_thresh, 2,
        np.where(y_train_agg < low_thresh, 0, 1)
    )
    y_test_lbl = np.where(
        y_test_agg > high_thresh, 2,
        np.where(y_test_agg < low_thresh, 0, 1)
    )

    # ---- Model A : aggression regressor --------------------------------- #
    model_a = XGBRegressor(**REGRESSOR_PARAMS)
    model_a.fit(X_train, y_train_agg)
    pred_agg = model_a.predict(X_test)

    mae_agg  = mean_absolute_error(y_test_agg, pred_agg)
    rmse_agg = float(np.sqrt(mean_squared_error(y_test_agg, pred_agg)))

    # ---- Model B : runs regressor --------------------------------------- #
    model_b = XGBRegressor(**REGRESSOR_PARAMS)
    model_b.fit(X_train, y_train_runs)
    pred_runs = model_b.predict(X_test)

    mae_runs  = mean_absolute_error(y_test_runs, pred_runs)
    rmse_runs = float(np.sqrt(mean_squared_error(y_test_runs, pred_runs)))

    # ---- Model C : H/M/L classifier ------------------------------------- #
    # Guard: if training set is missing any class, classifier will fail
    unique_classes = np.unique(y_train_lbl)
    if len(unique_classes) < 3:
        acc_lbl = np.nan
        f1_lbl  = np.nan
    else:
        model_c = XGBClassifier(**CLASSIFIER_PARAMS)
        model_c.fit(X_train, y_train_lbl)
        pred_lbl = model_c.predict(X_test)

        acc_lbl = float(np.mean(pred_lbl == y_test_lbl))
        f1_lbl  = f1_score(y_test_lbl, pred_lbl, average="macro", zero_division=0)

    return {
        "train_up_to":  train_up_to,
        "test_season":  test_season,
        "train_rows":   len(train_df),
        "test_rows":    len(test_df),
        "mae_agg":      round(mae_agg,  4),
        "rmse_agg":     round(rmse_agg, 4),
        "mae_runs":     round(mae_runs,  4),
        "rmse_runs":    round(rmse_runs, 4),
        "acc_label":    round(acc_lbl, 4) if not np.isnan(acc_lbl) else np.nan,
        "f1_label":     round(f1_lbl,  4) if not np.isnan(f1_lbl)  else np.nan,
    }


# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------

def evaluate(
    matrix_path: str  = str(DEFAULT_MATRIX),
    folds:       list = FOLDS,
    out_path:    "str | None" = None,
    verbose:     bool = True,
) -> pd.DataFrame:
    """
    Run all walk-forward folds and report results.

    Args:
        matrix_path : path to training_matrix.csv
        folds       : list of (train_up_to, test_season) tuples
        out_path    : if provided, save fold results to this CSV path
        verbose     : print results to console

    Returns:
        DataFrame with one row per fold plus a summary row
    """
    matrix_path = Path(matrix_path)
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Training matrix not found: {matrix_path}\n"
            f"Run:  python src/feature_builder.py"
        )

    df = pd.read_csv(matrix_path)
    df["season"] = pd.to_numeric(df["season"], errors="coerce")

    if verbose:
        seasons_present = sorted(df["season"].dropna().unique().astype(int))
        print(f"Loaded {len(df):,} rows  |  seasons: {seasons_present}")

    # Compute global thresholds from full dataset (used consistently across folds)
    agg_vals                       = df[TARGET_AGGRESSION].dropna()
    _, low_thresh, high_thresh     = make_labels(agg_vals)

    if verbose:
        print(f"Global thresholds  Low < {low_thresh:.2f}  |  High > {high_thresh:.2f}")
        print()

    # ---- Run folds ------------------------------------------------------- #
    results = []

    for train_up_to, test_season in folds:
        if verbose:
            print(f"Fold  train <= {train_up_to}  |  test = {test_season}", end="  ->  ")

        fold_result = _run_fold(
            df, train_up_to, test_season, low_thresh, high_thresh
        )

        if fold_result is None:
            if verbose:
                print()
            continue

        results.append(fold_result)

        if verbose:
            print(
                f"MAE agg={fold_result['mae_agg']:.2f}pp  "
                f"MAE runs={fold_result['mae_runs']:.2f}  "
                f"F1={fold_result['f1_label']:.3f}  "
                f"[{fold_result['train_rows']} train / {fold_result['test_rows']} test]"
            )

    if not results:
        print("No folds completed — check your season range in the matrix.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # ---- Summary row ----------------------------------------------------- #
    numeric_cols = ["mae_agg", "rmse_agg", "mae_runs", "rmse_runs",
                    "acc_label", "f1_label"]

    summary = {"train_up_to": "MEAN ± STD", "test_season": "", "train_rows": "", "test_rows": ""}
    for col in numeric_cols:
        vals = results_df[col].dropna()
        summary[col] = f"{vals.mean():.4f} ± {vals.std():.4f}"

    # ---- Print results table --------------------------------------------- #
    if verbose:
        print()
        print("=" * 75)
        print("WALK-FORWARD EVALUATION RESULTS")
        print("=" * 75)

        header = (
            f"{'Fold':<22} {'Train':>6} {'Test':>6} "
            f"{'MAE agg':>9} {'MAE runs':>9} {'F1':>7} {'Acc':>7}"
        )
        print(header)
        print("-" * 75)

        for r in results:
            fold_label = f"<={r['train_up_to']} -> {r['test_season']}"
            f1  = f"{r['f1_label']:.3f}"  if not np.isnan(r["f1_label"])  else "  N/A"
            acc = f"{r['acc_label']:.3f}" if not np.isnan(r["acc_label"]) else "  N/A"
            print(
                f"  {fold_label:<20} {r['train_rows']:>6} {r['test_rows']:>6} "
                f"{r['mae_agg']:>8.2f}% {r['mae_runs']:>8.2f}  "
                f"{f1:>7} {acc:>7}"
            )

        print("-" * 75)

        # Summary stats
        mae_agg_vals  = results_df["mae_agg"].dropna()
        mae_runs_vals = results_df["mae_runs"].dropna()
        f1_vals       = results_df["f1_label"].dropna()
        acc_vals      = results_df["acc_label"].dropna()

        print(
            f"  {'MEAN ± STD':<20} {'':>6} {'':>6} "
            f"{mae_agg_vals.mean():>7.2f}±{mae_agg_vals.std():.2f}% "
            f"{mae_runs_vals.mean():>7.2f}±{mae_runs_vals.std():.2f}  "
            f"{f1_vals.mean():>6.3f}±{f1_vals.std():.3f} "
            f"{acc_vals.mean():>6.3f}±{acc_vals.std():.3f}"
        )
        print("=" * 75)

        # Target assessment
        print("\nTarget assessment:")
        print(
            f"  MAE aggression  : {mae_agg_vals.mean():.2f}pp  "
            f"(target < 8pp)   {'PASS' if mae_agg_vals.mean() < 8 else 'FAIL'}"
        )
        print(
            f"  MAE runs        : {mae_runs_vals.mean():.2f}    "
            f"(target < 6)     {'PASS' if mae_runs_vals.mean() < 6 else 'FAIL'}"
        )
        print(
            f"  Macro F1        : {f1_vals.mean():.3f}   "
            f"(target > 0.60)  {'PASS' if f1_vals.mean() > 0.60 else 'FAIL'}"
        )
        print()
        print("These are your resume numbers — cite the mean ± std, not train metrics.")

    # ---- Save results ----------------------------------------------------- #
    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out, index=False)
        print(f"\nSaved fold results to: {out}")

    return results_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Walk-forward cross-validation for the IPL powerplay ML models."
    )
    ap.add_argument(
        "--matrix",
        default=str(DEFAULT_MATRIX),
        help=f"Path to training matrix CSV (default: {DEFAULT_MATRIX})",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Optional path to save fold results CSV",
    )
    args = ap.parse_args()

    evaluate(
        matrix_path = args.matrix,
        out_path    = args.out,
        verbose     = True,
    )