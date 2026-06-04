"""
explain_model.py
================
Generates SHAP-based feature explanations for the aggression regressor (Model A).

SHAP 0.49.1 cannot parse XGBoost's bracket+scientific notation base_score
format (e.g. '[1.3130446E1]'). This file patches the booster config before
passing it to TreeExplainer to work around the incompatibility.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODELS_DIR = Path("models")

# ---------------------------------------------------------------------------
# Human-readable feature names
# ---------------------------------------------------------------------------

FEATURE_DISPLAY_NAMES = {
    "bat_pp_aggression_prev1":    "Batting team aggression (prev season)",
    "bat_pp_aggression_prev2":    "Batting team aggression (2 seasons ago)",
    "bat_pp_aggression_2s_avg":   "Batting team 2-season weighted avg",
    "bat_pp_aggression_trend":    "Batting team aggression trend",
    "bat_pp_runs_prev1":          "Batting team avg PP runs (prev season)",
    "bat_pp_wicket_rate_prev1":   "Batting team wicket loss rate (prev season)",
    "bowl_pp_economy_prev1":      "Bowling team economy rate (prev season)",
    "bowl_pp_wicket_rate_prev1":  "Bowling team wicket rate (prev season)",
    "bowl_pp_economy_trend":      "Bowling team economy trend",
    "h2h_pp_aggression":          "Head-to-head PP aggression (historical)",
    "venue_pp_aggression":        "Venue avg PP aggression",
    "venue_pp_runs":              "Venue avg PP runs",
    "venue_innings_count":        "Venue data confidence (innings count)",
    "innings_num":                "Innings number (1st or 2nd)",
    "is_day_night":               "Day-Night match",
    "season_index":               "Season era index",
    "is_playoff":                 "Playoff match",
}

# ---------------------------------------------------------------------------
# Module-level cache — loaded once per process
# ---------------------------------------------------------------------------

_cache: dict = {}


def _patch_booster(booster):
    """
    Replace XGBoost's bracket+scientific notation base_score with a plain
    float string so SHAP 0.49.1 can parse it without crashing.

    XGBoost stores base_score as e.g. '[1.3130446E1]' or '[5E-1]'.
    SHAP calls float() on this directly and crashes.
    We overwrite it with '0.5' — a value SHAP can always parse.
    This does not affect model predictions; it only affects SHAP's internal
    baseline which we override anyway with predicted - sum(shap_values).
    """
    try:
        config = json.loads(booster.save_config())
        config["learner"]["learner_model_param"]["base_score"] = "0.5"
        booster.load_config(json.dumps(config))
    except Exception:
        pass  # if patching fails, let TreeExplainer try anyway
    return booster


def _load_explainer() -> dict:
    """Load model and build SHAP TreeExplainer. Cached after first call."""
    if _cache:
        return _cache

    model_path = MODELS_DIR / "regressor_aggression.pkl"
    meta_path  = MODELS_DIR / "meta.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run:  python src/train.py"
        )

    model = joblib.load(model_path)

    with open(meta_path) as f:
        meta = json.load(f)

    # Patch the booster before SHAP inspects it
    booster = _patch_booster(model.get_booster())

    # feature_perturbation="tree_path_dependent" is fast and exact,
    # and handles NaN features correctly without a background dataset
    explainer = shap.TreeExplainer(
        booster,
        feature_perturbation="tree_path_dependent",
    )

    _cache["model"]     = model
    _cache["explainer"] = explainer
    _cache["meta"]      = meta

    return _cache


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def explain_prediction(feature_row: pd.DataFrame, top_n: int = 6) -> dict:
    """
    Compute SHAP values for a single prediction row.

    Args:
        feature_row : single-row DataFrame from build_predict_features()
        top_n       : number of top features to return (by absolute SHAP value)

    Returns:
        dict with keys:
            baseline   float   mean predicted aggression across training data
            predicted  float   model's output aggression %
            features   list    top_n features sorted by abs(shap), each entry:
                                 name       human-readable feature name
                                 raw_name   original column name
                                 value      actual feature value (None if NaN)
                                 shap       SHAP contribution (+ pushed up)
                                 direction  "up" or "down"
    """
    cache     = _load_explainer()
    explainer = cache["explainer"]
    model     = cache["model"]

    # Ensure all values are float — int64 columns can cause dtype issues
    X = feature_row.copy().astype(float)

    # Compute SHAP values
    shap_values = explainer.shap_values(X)

    # Some SHAP/XGBoost versions wrap output in a list
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_arr = np.array(shap_values, dtype=float).flatten()

    # Prediction from the sklearn wrapper (consistent with predict.py)
    predicted = float(model.predict(feature_row)[0])
    predicted = max(0.0, min(100.0, predicted))

    # Derive baseline mathematically — avoids any expected_value parsing
    baseline = predicted - float(shap_arr.sum())

    # Build feature list
    feature_cols = feature_row.columns.tolist()
    feature_vals = feature_row.iloc[0].tolist()

    features = []
    for col, val, sv in zip(feature_cols, feature_vals, shap_arr):
        try:
            is_nan = bool(np.isnan(float(val)))
        except (TypeError, ValueError):
            is_nan = True

        features.append({
            "name":      FEATURE_DISPLAY_NAMES.get(col, col),
            "raw_name":  col,
            "value":     None if is_nan else round(float(val), 4),
            "shap":      round(float(sv), 4),
            "direction": "up" if sv >= 0 else "down",
        })

    # Sort by absolute impact descending
    features.sort(key=lambda x: abs(x["shap"]), reverse=True)

    return {
        "baseline":  round(baseline,  2),
        "predicted": round(predicted, 2),
        "features":  features[:top_n],
    }