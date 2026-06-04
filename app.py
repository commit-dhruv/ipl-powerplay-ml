"""
app.py
======
Flask web application for the IPL Powerplay Aggression ML model.

Three tabs:
    Predict     — select teams, venue, context -> get prediction + SHAP explanation
    Leaderboard — most aggressive teams historically by season
    Venues      — top powerplay venues ranked by aggression + usage tier

Usage
-----
    python app.py

Then open: http://localhost:5000
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, "src")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Home ground map
# ---------------------------------------------------------------------------

HOME_GROUNDS: dict[str, str] = {
    "Mumbai Indians":              "Wankhede Stadium",
    "Chennai Super Kings":         "MA Chidambaram Stadium, Chepauk",
    "Royal Challengers Bengaluru": "M Chinnaswamy Stadium",
    "Kolkata Knight Riders":       "Eden Gardens",
    "Delhi Capitals":              "Arun Jaitley Stadium",
    "Rajasthan Royals":            "Sawai Mansingh Stadium",
    "Sunrisers Hyderabad":         "Rajiv Gandhi International Stadium, Uppal",
    "Punjab Kings":                "Punjab Cricket Association Stadium, Mohali",
    "Gujarat Titans":              "Narendra Modi Stadium, Ahmedabad",
    "Lucknow Super Giants":        "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
}

# ---------------------------------------------------------------------------
# Venue usage tiers
# ---------------------------------------------------------------------------

TIER_REGULAR    = 30
TIER_OCCASIONAL = 15

def _usage_tier(innings_count: int) -> str:
    if innings_count >= TIER_REGULAR:
        return "Regular"
    elif innings_count >= TIER_OCCASIONAL:
        return "Occasional"
    return "Rare"

# ---------------------------------------------------------------------------
# Load static data at startup
# ---------------------------------------------------------------------------

PP_PATH    = Path("data/processed/ipl/powerplay_summary.csv")
MATCH_PATH = Path("data/processed/ipl/matches.csv")
META_PATH  = Path("models/meta.json")


def _load_app_data() -> dict:
    if not PP_PATH.exists() or not MATCH_PATH.exists():
        raise FileNotFoundError(
            "Processed data not found. Run the pipeline first:\n"
            "  python src/parse_cricsheet.py data/ipl --competition IPL --out data/processed/ipl\n"
            "  python src/feature_builder.py\n"
            "  python src/train.py"
        )

    pp      = pd.read_csv(PP_PATH,    parse_dates=["date"])
    matches = pd.read_csv(MATCH_PATH, parse_dates=["date"])
    pp["season"] = pd.to_numeric(pp["season"], errors="coerce")

    DEFUNCT = {
        "Deccan Chargers", "Kochi Tuskers Kerala",
        "Pune Warriors", "Gujarat Lions", "Rising Pune Supergiants",
    }
    all_teams = sorted(
        t for t in pp["batting_team"].dropna().unique()
        if t not in DEFUNCT
    )

    venue_counts = pp.groupby("venue")["pp_aggression"].count()
    valid_venues = sorted(venue_counts[venue_counts >= 5].index.tolist())

    with open(META_PATH) as f:
        meta = json.load(f)

    return {
        "pp":          pp,
        "matches":     matches,
        "teams":       all_teams,
        "venues":      valid_venues,
        "meta":        meta,
        "low_thresh":  meta["low_thresh"],
        "high_thresh": meta["high_thresh"],
    }


APP_DATA = _load_app_data()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_leaderboard(pp, season=None, min_innings=5):
    DEFUNCT = {
        "Deccan Chargers", "Kochi Tuskers Kerala",
        "Pune Warriors", "Gujarat Lions", "Rising Pune Supergiants",
    }
    data = pp.copy()
    if season is not None:
        data = data[data["season"] == season]
    data = data[~data["batting_team"].isin(DEFUNCT)]

    grp = (
        data.groupby("batting_team")
        .agg(
            mean_aggression=("pp_aggression", "mean"),
            mean_runs=("pp_runs", "mean"),
            mean_wickets=("pp_wickets", "mean"),
            innings_count=("pp_aggression", "count"),
        )
        .reset_index()
    )
    grp = grp[grp["innings_count"] >= min_innings]
    grp = grp.sort_values("mean_aggression", ascending=False).reset_index(drop=True)
    grp["rank"] = grp.index + 1

    low_t, high_t = APP_DATA["low_thresh"], APP_DATA["high_thresh"]
    grp["label"] = grp["mean_aggression"].apply(
        lambda v: "High" if v > high_t else ("Low" if v < low_t else "Medium")
    )
    return grp.round(2).to_dict(orient="records")


def _build_venue_table(pp, min_innings=5):
    grp = (
        pp.groupby("venue")
        .agg(
            mean_aggression=("pp_aggression", "mean"),
            mean_runs=("pp_runs", "mean"),
            innings_count=("pp_aggression", "count"),
        )
        .reset_index()
    )
    grp = grp[grp["innings_count"] >= min_innings]
    grp = grp.sort_values("mean_aggression", ascending=False).reset_index(drop=True)
    grp["rank"]       = grp.index + 1
    grp["usage_tier"] = grp["innings_count"].apply(_usage_tier)
    return grp.round(2).to_dict(orient="records")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    seasons = sorted(
        APP_DATA["pp"]["season"].dropna().unique().astype(int),
        reverse=True,
    )
    valid_set    = set(APP_DATA["venues"])
    home_grounds = {
        team: venue
        for team, venue in HOME_GROUNDS.items()
        if venue in valid_set
    }
    return render_template(
        "index.html",
        teams        = APP_DATA["teams"],
        venues       = APP_DATA["venues"],
        seasons      = seasons,
        home_grounds = json.dumps(home_grounds),
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    from predict import predict_both_innings

    data         = request.get_json()
    team_a       = data.get("team_a", "")
    team_b       = data.get("team_b", "")
    venue        = data.get("venue",  "")
    season       = int(data.get("season", 2026))
    is_day_night = bool(data.get("is_day_night", False))
    is_playoff   = bool(data.get("is_playoff",   False))

    if not team_a or not team_b or not venue:
        return jsonify({"error": "team_a, team_b and venue are required"}), 400
    if team_a == team_b:
        return jsonify({"error": "Team A and Team B must be different"}), 400

    try:
        inn1, inn2 = predict_both_innings(
            team_a=team_a, team_b=team_b, venue=venue,
            season=season, is_day_night=is_day_night,
            is_playoff=is_playoff, verbose=False,
        )
        diff = inn1["aggression_pct"] - inn2["aggression_pct"]
        return jsonify({
            "innings1":        inn1,
            "innings2":        inn2,
            "more_aggressive": team_a if diff > 0 else team_b,
            "diff":            round(abs(diff), 2),
            "low_thresh":      APP_DATA["low_thresh"],
            "high_thresh":     APP_DATA["high_thresh"],
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/explain", methods=["POST"])
def api_explain():
    """
    POST /api/explain
    Body (JSON): same as /api/predict — team_a, team_b, venue, season, ...

    Returns SHAP explanations for both innings.
    Called separately after /api/predict so the main result loads fast.
    """
    from feature_builder import build_predict_features
    from explain_model   import explain_prediction

    data         = request.get_json()
    team_a       = data.get("team_a", "")
    team_b       = data.get("team_b", "")
    venue        = data.get("venue",  "")
    season       = int(data.get("season", 2026))
    is_day_night = bool(data.get("is_day_night", False))
    is_playoff   = bool(data.get("is_playoff",   False))

    if not team_a or not team_b or not venue:
        return jsonify({"error": "team_a, team_b and venue are required"}), 400

    try:
        # Build feature vectors for both innings
        feat1 = build_predict_features(
            batting_team=team_a, bowling_team=team_b, venue=venue,
            innings=1, is_day_night=is_day_night,
            season=season, is_playoff=is_playoff,
        )
        feat2 = build_predict_features(
            batting_team=team_b, bowling_team=team_a, venue=venue,
            innings=2, is_day_night=is_day_night,
            season=season, is_playoff=is_playoff,
        )

        exp1 = explain_prediction(feat1, top_n=6)
        exp2 = explain_prediction(feat2, top_n=6)

        return jsonify({
            "innings1_team": team_a,
            "innings2_team": team_b,
            "innings1":      exp1,
            "innings2":      exp2,
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/leaderboard")
def api_leaderboard():
    season_param = request.args.get("season")
    season = int(season_param) if season_param else None
    rows = _build_leaderboard(APP_DATA["pp"], season=season)
    return jsonify(rows)


@app.route("/api/venues")
def api_venues():
    rows = _build_venue_table(APP_DATA["pp"])
    return jsonify(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting IPL Powerplay ML app...")
    print("Open: http://localhost:5000")
    app.run(debug=True, port=5000)