"""
feature_builder.py
==================
Builds the ML training matrix and prediction feature vectors from parsed
Cricsheet data.

Two public functions:

    build_training_matrix()
        Reads processed CSVs, engineers all features, returns a DataFrame
        where each row is one IPL powerplay innings (season >= 2016).
        Every feature uses only data from seasons BEFORE the match — no leakage.

    build_predict_features()
        Builds a single-row feature vector for a pre-match prediction.
        Call this at inference time from predict.py or app.py.

Typical usage
-------------
    from src.feature_builder import build_training_matrix, build_predict_features

    df = build_training_matrix()
    print(df.shape)          # (~1,080, 22)
    print(df.columns.tolist())

    row = build_predict_features(
        batting_team = "Mumbai Indians",
        bowling_team = "Chennai Super Kings",
        venue        = "Wankhede Stadium",
        innings      = 1,
        is_day_night = True,
        season       = 2026,
    )

CLI usage (generate and save the training matrix)
--------------------------------------------------
    python src/feature_builder.py --out data/processed/training_matrix.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------

IPL_PP_PATH      = Path("data/processed/ipl/powerplay_summary.csv")
IPL_MATCH_PATH   = Path("data/processed/ipl/matches.csv")
T20_PP_PATH      = Path("data/processed/t20s/powerplay_summary.csv")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Only build training rows from this season onwards.
# Earlier seasons still contribute to FEATURE computation.
MIN_TRAIN_SEASON = 2016

# Discard innings with fewer than this many legal balls in the powerplay.
# Handles rain-affected matches and abandoned games.
MIN_LEGAL_BALLS  = 18

# Overs in a full powerplay — used for economy/wicket-rate calculations
PP_OVERS = 6

# Season the model was built for — used in season_index feature
FIRST_IPL_SEASON = 2008


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_ipl_data() -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Load and lightly validate the processed IPL CSVs."""
    for p in [IPL_PP_PATH, IPL_MATCH_PATH]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing file: {p}\n"
                f"Run parse_cricsheet.py first:\n"
                f"  python src/parse_cricsheet.py data/ipl "
                f"--competition IPL --out data/processed/ipl"
            )

    pp = pd.read_csv(IPL_PP_PATH, parse_dates=["date"])
    mt = pd.read_csv(IPL_MATCH_PATH, parse_dates=["date"])

    pp["season"] = pd.to_numeric(pp["season"], errors="coerce").astype("Int64")
    mt["season"] = pd.to_numeric(mt["season"], errors="coerce").astype("Int64")

    # Drop super overs (innings 3+) and incomplete powerplays
    pp = pp[pp["innings"] <= 2].copy()
    pp = pp[pp["pp_legal_balls"] >= MIN_LEGAL_BALLS].copy()

    return pp, mt


def _load_t20_venue_data() -> "pd.DataFrame | None":
    """Load T20 powerplay summary for venue enrichment. Returns None if missing."""
    if not T20_PP_PATH.exists():
        print(
            f"  [INFO] T20 powerplay file not found at {T20_PP_PATH}. "
            f"Venue features will use IPL data only."
        )
        return None

    t20 = pd.read_csv(T20_PP_PATH, parse_dates=["date"])
    t20["season"] = pd.to_numeric(t20["season"], errors="coerce").astype("Int64")
    t20 = t20[t20["innings"] <= 2].copy()
    t20 = t20[t20["pp_legal_balls"] >= MIN_LEGAL_BALLS].copy()
    return t20


def _team_batting_stats(pp: pd.DataFrame) -> pd.DataFrame:
    """
    Compute batting-side powerplay stats per (batting_team, season).

    Returns one row per (team, season) with columns:
        bat_pp_aggression   mean aggression % across all innings
        bat_pp_runs         mean runs scored
        bat_pp_wickets      mean wickets lost
    """
    grp = (
        pp.groupby(["batting_team", "season"], as_index=False)
        .agg(
            bat_pp_aggression = ("pp_aggression", "mean"),
            bat_pp_runs       = ("pp_runs",       "mean"),
            bat_pp_wickets    = ("pp_wickets",    "mean"),
        )
    )
    return grp.rename(columns={"batting_team": "team"})


def _team_bowling_stats(pp: pd.DataFrame) -> pd.DataFrame:
    """
    Compute bowling-side powerplay stats per (bowling_team, season).

    Economy rate  = runs conceded per over  = pp_total_runs * 6 / pp_legal_balls
    Wicket rate   = wickets taken per over  = pp_wickets    * 6 / pp_legal_balls

    Returns one row per (team, season) with columns:
        bowl_pp_economy      mean economy rate as bowling team
        bowl_pp_wicket_rate  mean wicket rate as bowling team
    """
    tmp = pp.copy()

    # pp_total_runs and pp_wickets are from the BATTING team's perspective,
    # so they represent runs CONCEDED and wickets TAKEN by the bowling side.
    tmp["economy"]     = np.where(
        tmp["pp_legal_balls"] > 0,
        tmp["pp_total_runs"] * PP_OVERS / tmp["pp_legal_balls"],
        np.nan,
    )
    tmp["wicket_rate"] = np.where(
        tmp["pp_legal_balls"] > 0,
        tmp["pp_wickets"] * PP_OVERS / tmp["pp_legal_balls"],
        np.nan,
    )

    grp = (
        tmp.groupby(["bowling_team", "season"], as_index=False)
        .agg(
            bowl_pp_economy     = ("economy",     "mean"),
            bowl_pp_wicket_rate = ("wicket_rate", "mean"),
        )
    )
    return grp.rename(columns={"bowling_team": "team"})


def _venue_cumulative_stats(
    ipl_pp: pd.DataFrame,
    t20_pp: "pd.DataFrame | None",
) -> pd.DataFrame:
    """
    Compute cumulative venue powerplay stats per (venue, season).

    'Cumulative up to season N' means: all innings at this venue in seasons
    strictly LESS THAN N. This is the leakage-safe definition — when we join
    this to a match in season N, we only see history up to season N-1.

    T20 data is included to enrich venues that also host BBL, T20Is, etc.
    IPL data is weighted 1.0; non-IPL T20 data is included but IPL still
    dominates because we have more of it.

    Returns one row per (venue, season) with columns:
        venue_pp_aggression   cumulative mean aggression % at this venue
        venue_pp_runs         cumulative mean runs at this venue
        venue_innings_count   number of innings used (confidence proxy)
    """
    ipl_pp = ipl_pp.copy()
    ipl_pp["source"] = "ipl"

    if t20_pp is not None:
        t20_pp = t20_pp.copy()
        t20_pp["source"] = "t20"
        combined = pd.concat([ipl_pp, t20_pp], ignore_index=True)
    else:
        combined = ipl_pp

    # Drop rows with unknown venue
    combined = combined.dropna(subset=["venue"])

    # Get all (venue, season) pairs that appear in IPL (we only need venue
    # features for venues in our training data)
    ipl_venues  = ipl_pp["venue"].dropna().unique()
    all_seasons = sorted(ipl_pp["season"].dropna().unique())

    records = []
    for venue in ipl_venues:
        venue_data = combined[combined["venue"] == venue]

        for season in all_seasons:
            # Cumulative: all innings BEFORE this season
            hist = venue_data[venue_data["season"] < season]

            if len(hist) == 0:
                records.append({
                    "venue":                venue,
                    "season":               season,
                    "venue_pp_aggression":  np.nan,
                    "venue_pp_runs":        np.nan,
                    "venue_innings_count":  0,
                })
            else:
                records.append({
                    "venue":                venue,
                    "season":               season,
                    "venue_pp_aggression":  hist["pp_aggression"].mean(),
                    "venue_pp_runs":        hist["pp_runs"].mean(),
                    "venue_innings_count":  len(hist),
                })

    return pd.DataFrame(records)


def _h2h_cumulative_stats(pp: pd.DataFrame) -> pd.DataFrame:
    """
    Compute head-to-head powerplay aggression per (batting_team, bowling_team, season).

    Cumulative up to (but not including) the current season — same leakage-safe
    logic as venue stats.
    """
    pp = pp.dropna(subset=["batting_team", "bowling_team"])
    pairs   = pp[["batting_team", "bowling_team"]].drop_duplicates()
    seasons = sorted(pp["season"].dropna().unique())

    records = []
    for _, row in pairs.iterrows():
        bat  = row["batting_team"]
        bowl = row["bowling_team"]
        mask = (pp["batting_team"] == bat) & (pp["bowling_team"] == bowl)
        pair_data = pp[mask]

        for season in seasons:
            hist = pair_data[pair_data["season"] < season]
            records.append({
                "batting_team":       bat,
                "bowling_team":       bowl,
                "season":             season,
                "h2h_pp_aggression":  hist["pp_aggression"].mean() if len(hist) > 0 else np.nan,
                "h2h_innings_count":  len(hist),
            })

    return pd.DataFrame(records)


def _match_context(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Extract pre-match-knowable context features from matches.csv.

    is_day_night : True if the match had a day-night component
    is_playoff   : True for knockout/qualifier/final stages
    season_index : 1-based index (2008=1, 2009=2, ...) — captures era drift
    """
    mt = matches[["match_id", "match_stage"]].copy()

    # Day-night: Cricsheet doesn't always store this directly.
    # We derive it from match_number being null in later-stage matches, or
    # accept that we don't have this field and fill with 0.
    # If your matches.csv has a "day_night" column Cricsheet added, use that.
    if "day_night" in matches.columns:
        mt["is_day_night"] = matches["day_night"].notna().astype(int)
    else:
        mt["is_day_night"] = 0   # conservative default

    playoff_keywords = {
        "final", "qualifier", "eliminator",
        "semi final", "semi-final", "playoff"
    }

    def _is_playoff(stage) -> int:
        if pd.isna(stage):
            return 0
        return int(any(kw in str(stage).lower() for kw in playoff_keywords))

    mt["is_playoff"] = mt["match_stage"].apply(_is_playoff)

    return mt[["match_id", "is_day_night", "is_playoff"]]


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def build_training_matrix(
    min_train_season: int = MIN_TRAIN_SEASON,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build the full ML training matrix from processed IPL data.

    Every feature for a match in season N uses only data from seasons < N.
    No future data ever leaks into training rows.

    Args:
        min_train_season : earliest season to include as a training row.
                           Earlier seasons still feed the feature lookups.
        verbose          : print progress messages

    Returns:
        DataFrame with columns:
            Identifiers  : match_id, date, season, batting_team, bowling_team, venue
            Features (16): bat_*, bowl_*, h2h_*, venue_*, innings, is_day_night,
                           season_index, is_playoff
            Targets  (2) : pp_aggression, pp_runs
    """
    if verbose:
        print("Loading data...")

    pp, matches = _load_ipl_data()
    t20_pp      = _load_t20_venue_data()

    if verbose:
        print(f"  IPL innings      : {len(pp):,}")
        print(f"  IPL matches      : {len(matches):,}")
        if t20_pp is not None:
            print(f"  T20 innings      : {len(t20_pp):,}  (venue enrichment only)")

    # ---- Build lookup tables ------------------------------------------- #
    if verbose:
        print("\nBuilding lookup tables...")

    bat_stats   = _team_batting_stats(pp)
    bowl_stats  = _team_bowling_stats(pp)
    venue_stats = _venue_cumulative_stats(pp, t20_pp)
    h2h_stats   = _h2h_cumulative_stats(pp)
    ctx         = _match_context(matches)

    if verbose:
        print(f"  Batting stats    : {len(bat_stats):,} (team, season) pairs")
        print(f"  Bowling stats    : {len(bowl_stats):,} (team, season) pairs")
        print(f"  Venue stats      : {len(venue_stats):,} (venue, season) pairs")
        print(f"  H2H stats        : {len(h2h_stats):,} (bat, bowl, season) pairs")

    # ---- Filter to training rows --------------------------------------- #
    train = pp[pp["season"] >= min_train_season].copy()

    if verbose:
        print(f"\nTraining rows (season >= {min_train_season}): {len(train):,}")

    # Derive the lookup season: N-1 and N-2
    train["season_prev1"] = train["season"] - 1
    train["season_prev2"] = train["season"] - 2

    # ---- Join batting features (N-1) ----------------------------------- #
    train = train.merge(
        bat_stats.rename(columns={
            "team":              "batting_team",
            "bat_pp_aggression": "bat_pp_aggression_prev1",
            "bat_pp_runs":       "bat_pp_runs_prev1",
            "bat_pp_wickets":    "bat_pp_wicket_rate_prev1",
            "season":            "season_prev1",
        }),
        on=["batting_team", "season_prev1"],
        how="left",
    )

    # ---- Join batting features (N-2) for trend ------------------------- #
    train = train.merge(
        bat_stats.rename(columns={
            "team":              "batting_team",
            "bat_pp_aggression": "bat_pp_aggression_prev2",
            "season":            "season_prev2",
        })[["batting_team", "season_prev2", "bat_pp_aggression_prev2"]],
        on=["batting_team", "season_prev2"],
        how="left",
    )

    # Two-season average and trend (derived)
    train["bat_pp_aggression_2s_avg"] = (
        train["bat_pp_aggression_prev1"].fillna(0) * 0.6
        + train["bat_pp_aggression_prev2"].fillna(0) * 0.4
    )
    # Set to NaN if BOTH seasons are missing (truly no history)
    both_missing = (
        train["bat_pp_aggression_prev1"].isna()
        & train["bat_pp_aggression_prev2"].isna()
    )
    train.loc[both_missing, "bat_pp_aggression_2s_avg"] = np.nan

    train["bat_pp_aggression_trend"] = (
        train["bat_pp_aggression_prev1"] - train["bat_pp_aggression_prev2"]
    )

    # ---- Join bowling features (N-1) ----------------------------------- #
    train = train.merge(
        bowl_stats.rename(columns={
            "team":                 "bowling_team",
            "bowl_pp_economy":      "bowl_pp_economy_prev1",
            "bowl_pp_wicket_rate":  "bowl_pp_wicket_rate_prev1",
            "season":               "season_prev1",
        }),
        on=["bowling_team", "season_prev1"],
        how="left",
    )

    # ---- Join bowling features (N-2) for trend ------------------------- #
    train = train.merge(
        bowl_stats.rename(columns={
            "team":            "bowling_team",
            "bowl_pp_economy": "bowl_pp_economy_prev2",
            "season":          "season_prev2",
        })[["bowling_team", "season_prev2", "bowl_pp_economy_prev2"]],
        on=["bowling_team", "season_prev2"],
        how="left",
    )

    train["bowl_pp_economy_trend"] = (
        train["bowl_pp_economy_prev1"] - train["bowl_pp_economy_prev2"]
    )

    # ---- Join venue features ------------------------------------------- #
    train = train.merge(
        venue_stats.rename(columns={"season": "season"}),
        on=["venue", "season"],
        how="left",
    )

    # ---- Join H2H features --------------------------------------------- #
    train = train.merge(
        h2h_stats,
        on=["batting_team", "bowling_team", "season"],
        how="left",
    )

    # ---- Join match context -------------------------------------------- #
    train = train.merge(ctx, on="match_id", how="left")

    # ---- Derived context features -------------------------------------- #
    train["season_index"] = train["season"] - FIRST_IPL_SEASON + 1
    train["innings_num"]  = train["innings"].astype(int)

    # ---- Select and order final columns -------------------------------- #
    feature_cols = [
        # Batting team
        "bat_pp_aggression_prev1",
        "bat_pp_aggression_prev2",
        "bat_pp_aggression_2s_avg",
        "bat_pp_aggression_trend",
        "bat_pp_runs_prev1",
        "bat_pp_wicket_rate_prev1",
        # Bowling team
        "bowl_pp_economy_prev1",
        "bowl_pp_wicket_rate_prev1",
        "bowl_pp_economy_trend",
        # H2H
        "h2h_pp_aggression",
        # Venue
        "venue_pp_aggression",
        "venue_pp_runs",
        "venue_innings_count",
        # Context
        "innings_num",
        "is_day_night",
        "season_index",
        "is_playoff",
    ]

    target_cols = ["pp_aggression", "pp_runs"]

    id_cols = ["match_id", "date", "season", "batting_team", "bowling_team", "venue"]

    # Keep only columns that exist (safety for optional features)
    keep = [c for c in id_cols + feature_cols + target_cols if c in train.columns]
    result = train[keep].copy()

    # Drop rows where BOTH targets are NaN (shouldn't happen but safety check)
    result = result.dropna(subset=["pp_aggression", "pp_runs"], how="all")

    result = result.sort_values(["date", "match_id", "innings_num"]).reset_index(drop=True)

    if verbose:
        print(f"\nFinal training matrix : {result.shape[0]:,} rows x {result.shape[1]} cols")
        feature_nulls = result[feature_cols].isna().mean().round(3)
        print("\nNull rate per feature (expect some — NaN = no prior history):")
        for feat, null_rate in feature_nulls.items():
            bar = "█" * int(null_rate * 20)
            print(f"  {feat:<38} {null_rate:.1%}  {bar}")

    return result


# ---------------------------------------------------------------------------
# Prediction feature builder
# ---------------------------------------------------------------------------

def build_predict_features(
    batting_team:  str,
    bowling_team:  str,
    venue:         str,
    innings:       int,
    is_day_night:  bool,
    season:        int,
    is_playoff:    bool = False,
    verbose:       bool = False,
) -> pd.DataFrame:
    """
    Build a single-row feature vector for a pre-match prediction.

    Uses the same lookup tables as build_training_matrix() but for a
    specific upcoming match. All features are computed from seasons < season.

    Args:
        batting_team  : canonical team name (e.g. "Mumbai Indians")
        bowling_team  : canonical team name (e.g. "Chennai Super Kings")
        venue         : ground name matching Cricsheet (e.g. "Wankhede Stadium")
        innings       : 1 (batting first) or 2 (chasing)
        is_day_night  : True if match has a floodlit session
        season        : season being predicted (e.g. 2026)
        is_playoff    : True for knockout stage matches
        verbose       : print the feature vector for inspection

    Returns:
        Single-row DataFrame with the same 17 feature columns as the
        training matrix (no target columns).
    """
    pp, _ = _load_ipl_data()
    t20_pp = _load_t20_venue_data()

    bat_stats   = _team_batting_stats(pp)
    bowl_stats  = _team_bowling_stats(pp)
    venue_stats = _venue_cumulative_stats(pp, t20_pp)
    h2h_stats   = _h2h_cumulative_stats(pp)

    season_prev1 = season - 1
    season_prev2 = season - 2

    def _lookup_bat(col, s):
        row = bat_stats[(bat_stats["team"] == batting_team) & (bat_stats["season"] == s)]
        return float(row[col].iloc[0]) if len(row) > 0 else np.nan

    def _lookup_bowl(col, s):
        row = bowl_stats[(bowl_stats["team"] == bowling_team) & (bowl_stats["season"] == s)]
        return float(row[col].iloc[0]) if len(row) > 0 else np.nan
    
    # NEW — finds closest available season <= prediction season
    def _lookup_venue(col):
        candidates = venue_stats[
            (venue_stats["venue"] == venue) & (venue_stats["season"] <= season)
        ]
        if len(candidates) == 0:
            return np.nan
        row = candidates.loc[candidates["season"].idxmax()]
        return float(row[col])
    
    # NEW — finds closest available season <= prediction season
    def _lookup_h2h():
        candidates = h2h_stats[
            (h2h_stats["batting_team"] == batting_team)
            & (h2h_stats["bowling_team"] == bowling_team)
            & (h2h_stats["season"] <= season)
        ]
        if len(candidates) == 0:
            return np.nan
        row = candidates.loc[candidates["season"].idxmax()]
        return float(row["h2h_pp_aggression"])

    bat_prev1 = _lookup_bat("bat_pp_aggression", season_prev1)
    bat_prev2 = _lookup_bat("bat_pp_aggression", season_prev2)

    both_missing = np.isnan(bat_prev1) and np.isnan(bat_prev2)
    if both_missing:
        avg_2s = np.nan
    else:
        avg_2s = (
            (bat_prev1 if not np.isnan(bat_prev1) else 0) * 0.6
            + (bat_prev2 if not np.isnan(bat_prev2) else 0) * 0.4
        )

    features = {
        "bat_pp_aggression_prev1":    bat_prev1,
        "bat_pp_aggression_prev2":    bat_prev2,
        "bat_pp_aggression_2s_avg":   avg_2s,
        "bat_pp_aggression_trend":    bat_prev1 - bat_prev2 if not (np.isnan(bat_prev1) or np.isnan(bat_prev2)) else np.nan,
        "bat_pp_runs_prev1":          _lookup_bat("bat_pp_runs",    season_prev1),
        "bat_pp_wicket_rate_prev1":   _lookup_bat("bat_pp_wickets", season_prev1),
        "bowl_pp_economy_prev1":      _lookup_bowl("bowl_pp_economy",     season_prev1),
        "bowl_pp_wicket_rate_prev1":  _lookup_bowl("bowl_pp_wicket_rate", season_prev1),
        "bowl_pp_economy_trend":      (
            _lookup_bowl("bowl_pp_economy", season_prev1)
            - _lookup_bowl("bowl_pp_economy", season_prev2)
            if not (np.isnan(_lookup_bowl("bowl_pp_economy", season_prev1)) or
                    np.isnan(_lookup_bowl("bowl_pp_economy", season_prev2)))
            else np.nan
        ),
        "h2h_pp_aggression":          _lookup_h2h(),
        "venue_pp_aggression":        _lookup_venue("venue_pp_aggression"),
        "venue_pp_runs":              _lookup_venue("venue_pp_runs"),
        "venue_innings_count":        _lookup_venue("venue_innings_count"),
        "innings_num":                innings,
        "is_day_night":               int(is_day_night),
        "season_index":               season - FIRST_IPL_SEASON + 1,
        "is_playoff":                 int(is_playoff),
    }

    row_df = pd.DataFrame([features])

    if verbose:
        print(f"\nFeature vector — {batting_team} vs {bowling_team} at {venue} (season {season})")
        for k, v in features.items():
            val = f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else str(v)
            print(f"  {k:<38} {val}")

    return row_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Build the ML training matrix from processed IPL data."
    )
    ap.add_argument(
        "--out",
        default="data/processed/training_matrix.csv",
        help="Output path for training matrix CSV (default: data/processed/training_matrix.csv)",
    )
    ap.add_argument(
        "--min-season",
        type=int,
        default=MIN_TRAIN_SEASON,
        help=f"Earliest season to include as a training row (default: {MIN_TRAIN_SEASON})",
    )
    args = ap.parse_args()

    df = build_training_matrix(min_train_season=args.min_season, verbose=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\nSaved training matrix to: {out}")
    print("\n--- Sample (first 3 rows, key columns) ---")
    show_cols = [
        "season", "batting_team", "bowling_team",
        "bat_pp_aggression_prev1", "bowl_pp_economy_prev1",
        "venue_pp_aggression", "pp_aggression", "pp_runs",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    print(df[show_cols].head(3).to_string(index=False))