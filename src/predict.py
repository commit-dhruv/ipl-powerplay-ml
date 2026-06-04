"""
predict.py
==========
Loads trained models and produces a pre-match powerplay prediction
for a given team matchup and venue.

Output for each team:
    aggression_pct  predicted powerplay aggression % (Model A)
    pp_runs         predicted powerplay runs          (Model B)
    label           High / Medium / Low               (derived from Model A)

H/M/L label is derived directly from Model A's continuous output using
the thresholds stored in models/meta.json — no separate classifier needed.

Usage (CLI)
-----------
    python src/predict.py \
        --batting  "Mumbai Indians" \
        --bowling  "Chennai Super Kings" \
        --venue    "Wankhede Stadium" \
        --innings  1 \
        --season   2026

Usage (from another script)
----------------------------
    from src.predict import predict_match

    result = predict_match(
        batting_team = "Mumbai Indians",
        bowling_team = "Chennai Super Kings",
        venue        = "Wankhede Stadium",
        innings      = 1,
        season       = 2026,
        is_day_night = True,
        is_playoff   = False,
    )
    print(result)
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Allow running directly as script from project root
sys.path.insert(0, str(Path(__file__).parent))
from feature_builder import build_predict_features

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODELS_DIR = Path("models")


# ---------------------------------------------------------------------------
# Load models (cached at module level so repeated calls don't reload)
# ---------------------------------------------------------------------------

_cache: dict = {}


def _load_models() -> dict:
    """
    Load all models and metadata from models/ folder.
    Results are cached after the first call.
    """
    if _cache:
        return _cache

    meta_path = MODELS_DIR / "meta.json"
    path_a    = MODELS_DIR / "regressor_aggression.pkl"
    path_b    = MODELS_DIR / "regressor_runs.pkl"

    for p in [meta_path, path_a, path_b]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing model file: {p}\n"
                f"Run first:  python src/train.py"
            )

    with open(meta_path) as f:
        meta = json.load(f)

    _cache["model_a"]     = joblib.load(path_a)
    _cache["model_b"]     = joblib.load(path_b)
    _cache["low_thresh"]  = meta["low_thresh"]
    _cache["high_thresh"] = meta["high_thresh"]
    _cache["meta"]        = meta

    return _cache


# ---------------------------------------------------------------------------
# Label derivation
# ---------------------------------------------------------------------------

def score_to_label(
    aggression_pct: float,
    low_thresh:     float,
    high_thresh:    float,
) -> str:
    """
    Convert a continuous aggression % into a H/M/L label.

    Thresholds come from models/meta.json — computed from the full
    training dataset during train.py so they are always consistent.

    Args:
        aggression_pct : predicted aggression % (0-100 scale)
        low_thresh     : aggression below this  -> Low
        high_thresh    : aggression above this  -> High

    Returns:
        "High", "Medium", or "Low"
    """
    if aggression_pct > high_thresh:
        return "High"
    elif aggression_pct < low_thresh:
        return "Low"
    else:
        return "Medium"


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------

def predict_match(
    batting_team:  str,
    bowling_team:  str,
    venue:         str,
    innings:       int   = 1,
    season:        int   = 2026,
    is_day_night:  bool  = False,
    is_playoff:    bool  = False,
    verbose:       bool  = False,
) -> dict:
    """
    Produce a powerplay prediction for one team's innings.

    Args:
        batting_team  : team batting in the powerplay (canonical name)
        bowling_team  : team bowling in the powerplay
        venue         : ground name matching Cricsheet format
        innings       : 1 (batting first) or 2 (chasing)
        season        : season being predicted (e.g. 2026)
        is_day_night  : True if the match has a floodlit session
        is_playoff    : True for knockout / qualifier / final
        verbose       : print the prediction to console

    Returns:
        dict with keys:
            batting_team    str
            bowling_team    str
            venue           str
            innings         int
            aggression_pct  float   predicted powerplay aggression %
            pp_runs         float   predicted powerplay runs
            label           str     "High" / "Medium" / "Low"
            low_thresh      float   threshold used for Low boundary
            high_thresh     float   threshold used for High boundary
            null_features   list    features that were NaN (no prior history)
    """
    models = _load_models()

    # Build feature vector
    feature_row = build_predict_features(
        batting_team = batting_team,
        bowling_team = bowling_team,
        venue        = venue,
        innings      = innings,
        is_day_night = is_day_night,
        season       = season,
        is_playoff   = is_playoff,
        verbose      = verbose,
    )

    # Track which features have no history (NaN)
    null_features = [
        col for col in feature_row.columns
        if feature_row[col].isna().any()
    ]

    # Predict — XGBoost handles NaN natively
    aggression_pct = float(models["model_a"].predict(feature_row)[0])
    pp_runs        = float(models["model_b"].predict(feature_row)[0])

    # Clamp to sensible range (model can rarely extrapolate slightly outside)
    aggression_pct = max(0.0, min(100.0, aggression_pct))
    pp_runs        = max(0.0, pp_runs)

    label = score_to_label(
        aggression_pct,
        models["low_thresh"],
        models["high_thresh"],
    )

    result = {
        "batting_team":   batting_team,
        "bowling_team":   bowling_team,
        "venue":          venue,
        "innings":        innings,
        "aggression_pct": round(aggression_pct, 2),
        "pp_runs":        round(pp_runs, 1),
        "label":          label,
        "low_thresh":     round(models["low_thresh"],  2),
        "high_thresh":    round(models["high_thresh"], 2),
        "null_features":  null_features,
    }

    if verbose:
        _print_result(result)

    return result


def predict_both_innings(
    team_a:       str,
    team_b:       str,
    venue:        str,
    season:       int  = 2026,
    is_day_night: bool = False,
    is_playoff:   bool = False,
    verbose:      bool = True,
) -> "tuple[dict, dict]":
    """
    Predict powerplay for both innings of a match.

    Innings 1: team_a bats, team_b bowls.
    Innings 2: team_b bats, team_a bowls.

    Returns:
        (innings1_result, innings2_result) — both are dicts from predict_match()
    """
    innings1 = predict_match(
        batting_team = team_a,
        bowling_team = team_b,
        venue        = venue,
        innings      = 1,
        season       = season,
        is_day_night = is_day_night,
        is_playoff   = is_playoff,
        verbose      = False,
    )

    innings2 = predict_match(
        batting_team = team_b,
        bowling_team = team_a,
        venue        = venue,
        innings      = 2,
        season       = season,
        is_day_night = is_day_night,
        is_playoff   = is_playoff,
        verbose      = False,
    )

    if verbose:
        _print_match_summary(team_a, team_b, venue, innings1, innings2)

    return innings1, innings2


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

LABEL_COLOURS = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}


def _print_result(r: dict) -> None:
    """Print a single innings prediction."""
    icon = LABEL_COLOURS.get(r["label"], "")
    print(
        f"\n  {r['batting_team']} vs {r['bowling_team']}"
        f"  |  Innings {r['innings']}  |  {r['venue']}"
    )
    print(f"  Aggression   : {r['aggression_pct']:.2f}%  {icon} {r['label']}")
    print(f"  Predicted PP runs : {r['pp_runs']:.1f}")
    print(f"  Thresholds   : Low < {r['low_thresh']}  |  High > {r['high_thresh']}")
    if r["null_features"]:
        print(f"  [!] No history for: {', '.join(r['null_features'])}")


def _print_match_summary(
    team_a:   str,
    team_b:   str,
    venue:    str,
    innings1: dict,
    innings2: dict,
) -> None:
    """Print a two-innings match summary."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  POWERPLAY PREDICTION")
    print(f"  {team_a}  vs  {team_b}")
    print(f"  {venue}")
    print(sep)

    for r in [innings1, innings2]:
        icon = LABEL_COLOURS.get(r["label"], "")
        innings_label = "1st innings" if r["innings"] == 1 else "2nd innings"
        print(
            f"\n  {innings_label} — {r['batting_team']} batting"
        )
        print(f"  Aggression   : {r['aggression_pct']:.2f}%  {icon} {r['label']}")
        print(f"  Predicted PP : {r['pp_runs']:.1f} runs")
        if r["null_features"]:
            print(f"  [!] Missing history: {', '.join(r['null_features'])}")

    # Comparative verdict
    diff = innings1["aggression_pct"] - innings2["aggression_pct"]
    more_aggressive = team_a if diff > 0 else team_b
    print(f"\n  Verdict : {more_aggressive} expected to be more aggressive "
          f"in the powerplay (+{abs(diff):.2f}%)")
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Predict IPL powerplay aggression for a given matchup."
    )
    ap.add_argument("--batting",      required=True, help="Batting team name")
    ap.add_argument("--bowling",      required=True, help="Bowling team name")
    ap.add_argument("--venue",        required=True, help="Ground name")
    ap.add_argument("--innings",      type=int, default=1,
                    help="Innings number: 1 or 2 (default: 1)")
    ap.add_argument("--season",       type=int, default=2026,
                    help="Season being predicted (default: 2026)")
    ap.add_argument("--day-night",    action="store_true",
                    help="Flag if match is a day-night game")
    ap.add_argument("--playoff",      action="store_true",
                    help="Flag if match is a playoff game")
    ap.add_argument("--both-innings", action="store_true",
                    help="Predict both innings (ignores --innings flag)")

    args = ap.parse_args()

    if args.both_innings:
        predict_both_innings(
            team_a       = args.batting,
            team_b       = args.bowling,
            venue        = args.venue,
            season       = args.season,
            is_day_night = args.day_night,
            is_playoff   = args.playoff,
            verbose      = True,
        )
    else:
        predict_match(
            batting_team = args.batting,
            bowling_team = args.bowling,
            venue        = args.venue,
            innings      = args.innings,
            season       = args.season,
            is_day_night = args.day_night,
            is_playoff   = args.playoff,
            verbose      = True,
        )