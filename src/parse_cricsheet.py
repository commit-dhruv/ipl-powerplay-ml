"""
parse_cricsheet.py
==================
Parses Cricsheet JSON match files into structured DataFrames.

Supports THREE input layouts automatically:
    1. A folder of .json files          data/ipl/
    2. A single .zip file in folder     data/ipl/ipl.zip
    3. A folder of multiple .zip files  data/t20s/

Three outputs:
    balls_df           one row per delivery
    matches_df         one row per match (metadata)
    powerplay_df       one row per innings, powerplay stats aggregated

Typical usage
-------------
    from src.parse_cricsheet import parse_folder, make_powerplay_summary

    balls_df, matches_df = parse_folder("data/ipl", competition="IPL")
    pp_df = make_powerplay_summary(balls_df)

    balls_t20, matches_t20 = parse_folder("data/t20s")

CLI usage
---------
    python src/parse_cricsheet.py data/ipl --competition IPL --out data/processed/ipl
    python src/parse_cricsheet.py data/t20s --out data/processed/t20s
"""

import json
import zipfile
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from team_names import normalise_team
from team_names import normalise_team, normalise_venue


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Overs 0-5 (0-indexed in Cricsheet) = overs 1-6 in cricket = full powerplay
POWERPLAY_OVERS = set(range(6))


# ---------------------------------------------------------------------------
# Season normalisation
# ---------------------------------------------------------------------------

def normalise_season(raw) -> "int | None":
    """
    Convert Cricsheet season strings to a single integer year.

        "2007/08"  -> 2008
        "2023/24"  -> 2024
        "2023"     -> 2023
        "2023-24"  -> 2024

    Always returns the LATER year.
    """
    raw = str(raw).strip()

    for sep in ("/", "-"):
        if sep in raw:
            parts = raw.split(sep)
            try:
                base   = int(parts[0])
                suffix = parts[1].strip()
                if len(suffix) == 2:
                    return (base // 100) * 100 + int(suffix)
                else:
                    return int(suffix)
            except (ValueError, IndexError):
                return None

    try:
        return int(raw)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Core parser — works on an already-loaded dict
# ---------------------------------------------------------------------------

def _parse_json_dict(data: dict, match_id: str) -> "tuple[list[dict], dict | None]":
    """
    Parse one match dict (loaded from JSON) into delivery rows + metadata.
    Works identically whether the source is a .json file or entry in a .zip.
    """
    info    = data.get("info", {})
    event   = info.get("event", {})
    toss    = info.get("toss", {})
    outcome = info.get("outcome", {})

    dates    = info.get("dates", [])
    date_str = dates[0] if dates else None

    season = normalise_season(info.get("season", ""))

    teams  = info.get("teams", [])
    team1  = teams[0] if len(teams) > 0 else None
    team2  = teams[1] if len(teams) > 1 else None

    by             = outcome.get("by", {})
    competition    = event.get("name", "Unknown")

    match_meta = {
        "match_id":       match_id,
        "date":           date_str,
        "season":         season,
        "competition":    competition,
        "team1":          team1,
        "team2":          team2,
        "venue":          info.get("venue",       None),
        "city":           info.get("city",        None),
        "toss_winner":    toss.get("winner",      None),
        "toss_decision":  toss.get("decision",    None),   # "bat" or "field"
        "winner":         outcome.get("winner",   None),
        "win_by_runs":    by.get("runs",          None),
        "win_by_wickets": by.get("wickets",       None),
        "match_stage":    event.get("stage",      None),
        "match_number":   event.get("match_number", None),
        "gender":         info.get("gender",      None),
        "match_type":     info.get("match_type",  None),
    }

    venue      = info.get("venue", None)
    city       = info.get("city",  None)
    deliveries: list[dict] = []

    for innings_idx, innings in enumerate(data.get("innings", [])):
        batting_team = innings.get("team")

        if batting_team and len(teams) == 2:
            bowling_team = teams[1] if batting_team == teams[0] else teams[0]
        else:
            bowling_team = None

        innings_num = innings_idx + 1   # 3/4 = super over, filtered later

        for over_data in innings.get("overs", []):
            over_num     = over_data.get("over", 0)   # 0-indexed in Cricsheet
            is_powerplay = over_num in POWERPLAY_OVERS

            for ball_idx, delivery in enumerate(over_data.get("deliveries", [])):
                runs        = delivery.get("runs", {})
                batter_runs = runs.get("batter", 0)
                extras_runs = runs.get("extras", 0)
                total_runs  = runs.get("total",  0)

                extras    = delivery.get("extras", {})
                is_wide   = "wides"   in extras
                is_noball = "noballs" in extras
                is_legbye = "legbyes" in extras
                is_bye    = "byes"    in extras

                # Legal ball: not a wide, not a no-ball
                is_legal = not is_wide and not is_noball

                # Boundary: batter_runs == 4/6 AND not a wide extra
                is_four = (batter_runs == 4) and not is_wide
                is_six  = (batter_runs == 6) and not is_wide

                # Wickets: key may be absent OR an empty list
                wickets     = delivery.get("wickets", [])
                is_wicket   = len(wickets) > 0
                wicket_kind = wickets[0].get("kind") if is_wicket else None

                deliveries.append({
                    # Identifiers
                    "match_id":      match_id,
                    "date":          date_str,
                    "season":        season,
                    "competition":   competition,
                    "venue":         venue,
                    "city":          city,
                    # Teams
                    "batting_team":  batting_team,
                    "bowling_team":  bowling_team,
                    # Position
                    "innings":       innings_num,
                    "over":          over_num,
                    "ball":          ball_idx + 1,
                    "is_powerplay":  is_powerplay,
                    # Players
                    "batter":        delivery.get("batter"),
                    "bowler":        delivery.get("bowler"),
                    # Runs
                    "batter_runs":   batter_runs,
                    "extras_runs":   extras_runs,
                    "total_runs":    total_runs,
                    # Delivery flags
                    "is_wide":       is_wide,
                    "is_noball":     is_noball,
                    "is_legbye":     is_legbye,
                    "is_bye":        is_bye,
                    "is_legal":      is_legal,
                    # Outcomes
                    "is_four":       is_four,
                    "is_six":        is_six,
                    "is_wicket":     is_wicket,
                    "wicket_kind":   wicket_kind,
                })

    return deliveries, match_meta


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_json_file(filepath: str) -> "tuple[list[dict], dict | None]":
    """Load and parse a standalone .json file."""
    match_id = Path(filepath).stem
    try:
        with open(filepath, encoding="utf-8") as fh:
            data = json.load(fh)
        return _parse_json_dict(data, match_id)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [SKIP] {match_id}: {exc}")
        return [], None


def _load_zip(zip_path: str) -> "tuple[list[dict], list[dict]]":
    """
    Read every .json inside a .zip archive without extracting it to disk.
    Uses Python's built-in zipfile module — no extra dependency needed.
    """
    all_deliveries: list[dict] = []
    all_matches:    list[dict] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        json_names = [n for n in zf.namelist() if n.endswith(".json")]

        for name in json_names:
            match_id = Path(name).stem
            try:
                raw  = zf.read(name)
                data = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as exc:
                print(f"  [SKIP] {match_id}: {exc}")
                continue

            deliveries, match_meta = _parse_json_dict(data, match_id)
            if match_meta is not None:
                all_deliveries.extend(deliveries)
                all_matches.append(match_meta)

    return all_deliveries, all_matches


# ---------------------------------------------------------------------------
# Auto-detecting folder parser — the main public function
# ---------------------------------------------------------------------------

def parse_folder(
    folder_path: str,
    competition: "str | None" = None,
    verbose: bool = True,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """
    Parse Cricsheet data from a folder. Detects the layout automatically:

        Layout 1 — folder of .json files  -> reads each .json directly
        Layout 2 — one .zip in folder     -> reads JSONs inside it
        Layout 3 — multiple .zips         -> reads all zips in the folder

    Args:
        folder_path  : path to folder (e.g. "data/ipl" or "data/t20s")
        competition  : optional string to overwrite every row's competition
                       field. Use "IPL" to normalise across all IPL seasons.
        verbose      : whether to print progress

    Returns:
        balls_df     : ball-by-ball DataFrame
        matches_df   : match metadata DataFrame
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    json_files = sorted(folder.glob("*.json"))
    zip_files  = sorted(folder.glob("*.zip"))

    if not json_files and not zip_files:
        raise FileNotFoundError(
            f"No .json or .zip files found in: {folder_path}\n"
            f"Contents: {[f.name for f in folder.iterdir()]}"
        )

    all_deliveries: list[dict] = []
    all_matches:    list[dict] = []

    # Layout 1 — raw JSON files
    if json_files:
        if verbose:
            print(f"Layout: {len(json_files):,} .json files  [{folder_path}]")
        for i, fp in enumerate(json_files, 1):
            if verbose and i % 500 == 0:
                print(f"  {i:,}/{len(json_files):,} parsed...")
            deliveries, meta = _load_json_file(str(fp))
            if meta is not None:
                all_deliveries.extend(deliveries)
                all_matches.append(meta)

    # Layout 2 / 3 — zip files
    else:
        if verbose:
            print(f"Layout: {len(zip_files)} .zip file(s)  [{folder_path}]")
        for zp in zip_files:
            if verbose:
                print(f"  Reading {zp.name} ...")
            deliveries, matches = _load_zip(str(zp))
            all_deliveries.extend(deliveries)
            all_matches.extend(matches)
            if verbose:
                print(f"    -> {len(matches):,} matches, {len(deliveries):,} deliveries")

    # Override competition name if requested
    if competition is not None:
        for d in all_deliveries:
            d["competition"] = competition
        for m in all_matches:
            m["competition"] = competition

    # Build DataFrames
    balls_df   = pd.DataFrame(all_deliveries)
    matches_df = pd.DataFrame(all_matches)

    # Type casting
    if not balls_df.empty:
        balls_df["date"]   = pd.to_datetime(balls_df["date"], errors="coerce")
        balls_df["season"] = balls_df["season"].astype("Int64")

        for col in ["is_powerplay", "is_wide", "is_noball", "is_legbye",
                    "is_bye", "is_legal", "is_four", "is_six", "is_wicket"]:
            if col in balls_df.columns:
                balls_df[col] = balls_df[col].astype(bool)

        for col in ["over", "ball", "innings",
                    "batter_runs", "extras_runs", "total_runs"]:
            if col in balls_df.columns:
                balls_df[col] = balls_df[col].astype("Int64")

    if not matches_df.empty:
        matches_df["date"]   = pd.to_datetime(matches_df["date"], errors="coerce")
        matches_df["season"] = matches_df["season"].astype("Int64")

    # Normalise franchise name changes
    for col in ["batting_team", "bowling_team", "team1", "team2"]:
        if col in balls_df.columns:
            balls_df[col] = balls_df[col].map(normalise_team)
    for col in ["team1", "team2", "toss_winner", "winner"]:
        if col in matches_df.columns:
            matches_df[col] = matches_df[col].map(normalise_team)

    # Normalise venue name variants
    from team_names import normalise_venue
    if "venue" in balls_df.columns:
        balls_df["venue"] = balls_df["venue"].map(normalise_venue)
    if "venue" in matches_df.columns:
        matches_df["venue"] = matches_df["venue"].map(normalise_venue)

    if verbose:
        print(
            f"\nDone.  "
            f"{len(all_matches):,} matches  |  "
            f"{len(all_deliveries):,} deliveries"
        )

    return balls_df, matches_df


# ---------------------------------------------------------------------------
# Powerplay summary
# ---------------------------------------------------------------------------

def make_powerplay_summary(balls_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ball-by-ball data to one row per innings (powerplay only).

    Aggression formula:
        pp_aggression = (sixes * 1.0 + fours * 0.5) / legal_balls * 100
    """
    if balls_df.empty:
        return pd.DataFrame()

    pp = balls_df[balls_df["is_powerplay"]].copy()
    if pp.empty:
        return pd.DataFrame()

    group_cols = [
        "match_id", "date", "season", "competition",
        "batting_team", "bowling_team", "innings", "venue", "city",
    ]

    agg = (
        pp.groupby(group_cols, as_index=False)
        .agg(
            pp_runs        =("batter_runs",  "sum"),
            pp_total_runs  =("total_runs",   "sum"),
            pp_fours       =("is_four",      "sum"),
            pp_sixes       =("is_six",       "sum"),
            pp_wickets     =("is_wicket",    "sum"),
            pp_legal_balls =("is_legal",     "sum"),
        )
    )

    agg["pp_aggression"] = np.where(
        agg["pp_legal_balls"] > 0,
        (agg["pp_sixes"] * 1.0 + agg["pp_fours"] * 0.5)
            / agg["pp_legal_balls"] * 100,
        np.nan,
    )
    agg["pp_aggression"] = agg["pp_aggression"].round(4)

    return agg.sort_values(["date", "match_id", "innings"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(balls_df: pd.DataFrame, matches_df: pd.DataFrame) -> None:
    """Print a quick sanity-check report after parsing."""
    print("\n" + "=" * 55)
    print("VALIDATION REPORT")
    print("=" * 55)

    print(f"\nballs_df   : {balls_df.shape[0]:>8,} rows x {balls_df.shape[1]} cols")
    print(f"matches_df : {matches_df.shape[0]:>8,} rows x {matches_df.shape[1]} cols")

    # Null warnings on critical columns
    for col in ["date", "season", "batting_team", "bowling_team", "venue"]:
        if col in balls_df.columns:
            n = balls_df[col].isna().sum()
            if n > 0:
                print(f"  [WARN] balls_df['{col}'] has {n:,} nulls")

    if "season" in matches_df.columns:
        seasons = sorted(matches_df["season"].dropna().unique())
        print(f"\nSeasons covered : {list(seasons)}")
        print("\nMatches per season:")
        print(matches_df["season"].value_counts().sort_index().to_string())

    if "batting_team" in balls_df.columns:
        teams = sorted(balls_df["batting_team"].dropna().unique())
        print(f"\nUnique teams ({len(teams)}):")
        for t in teams:
            print(f"  {t}")

    if "venue" in matches_df.columns:
        print(f"\nMatches missing venue : {matches_df['venue'].isna().sum()}")

    dupes = matches_df["match_id"].duplicated().sum()
    print(f"Duplicate match IDs   : {dupes}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Parse Cricsheet JSON/.zip files into DataFrames."
    )
    ap.add_argument("folder",
                    help="Folder with .json or .zip files (e.g. data/ipl)")
    ap.add_argument("--competition", default=None,
                    help="Override competition name for all rows (e.g. 'IPL')")
    ap.add_argument("--out", default="data/processed",
                    help="Output folder for CSVs (default: data/processed)")
    args = ap.parse_args()

    balls_df, matches_df = parse_folder(
        args.folder,
        competition=args.competition,
        verbose=True,
    )

    pp_df = make_powerplay_summary(balls_df)
    validate(balls_df, matches_df)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    balls_df.to_csv(out / "balls.csv",             index=False)
    matches_df.to_csv(out / "matches.csv",         index=False)
    pp_df.to_csv(out / "powerplay_summary.csv",    index=False)

    print(f"\nSaved to: {out}/")
    print("  balls.csv")
    print("  matches.csv")
    print("  powerplay_summary.csv")

    print("\n--- Powerplay summary sample (first 5 rows) ---")
    print(pp_df.head(5).to_string(index=False))