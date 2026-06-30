"""Pull starting pitcher season stats (2015-2024) using Baseball-Reference data.

Mirrors the hitter project's approach: bwar_pitch() is a static data file
(not a scraped page), pitching_stats_bref(season) is looped per-season.

Usage:
    python -m src.data.pull_pitching_stats
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from pybaseball import bwar_pitch, pitching_stats_bref

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

START_SEASON = 2015
END_SEASON = 2024


def pull_war_data(start: int, end: int) -> pd.DataFrame:
    """Pull bwar_pitch, filter to our season range, aggregate multi-stint seasons."""
    print("Pulling bwar_pitch() — full history WAR table (one request)...")
    war = bwar_pitch()
    print(f"  Raw: {len(war):,} rows")
    print(f"  Columns: {war.columns.tolist()}")

    war = war[(war["year_ID"] >= start) & (war["year_ID"] <= end)].copy()
    print(f"  After season filter: {len(war):,} rows")

    agg = (
        war.groupby(["mlb_ID", "year_ID"], as_index=False)
        .agg(
            name_common=("name_common", "first"),
            player_ID=("player_ID", "first"),
            G=("G", "sum"),
            GS=("GS", "sum"),
            RA=("RA", "sum"),
            xRA=("xRA", "sum"),
            WAR=("WAR", "sum"),
            WAA=("WAA", "sum"),
        )
    )
    print(f"  After stint aggregation: {len(agg):,} player-seasons")
    return agg


def pull_rate_stats(start: int, end: int) -> pd.DataFrame:
    """Loop pitching_stats_bref over each season."""
    frames = []
    for year in range(start, end + 1):
        print(f"  Pulling pitching_stats_bref({year})...")
        df = pitching_stats_bref(year)
        df["year_ID"] = year
        frames.append(df)
        time.sleep(1.0)

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Combined: {len(combined):,} rows across {end - start + 1} seasons")
    return combined


def fix_encoding(df: pd.DataFrame, col: str) -> pd.Series:
    """Fix literal-escape-sequence names (same issue as the hitter project)."""
    def _fix(val):
        if not isinstance(val, str):
            return val
        if "\\x" not in val:
            return val
        try:
            return val.encode("utf-8").decode("unicode_escape").encode("latin1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return val
    return df[col].apply(_fix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=START_SEASON)
    parser.add_argument("--end", type=int, default=END_SEASON)
    parser.add_argument(
        "--out",
        type=Path,
        default=RAW_DIR / f"pitchers_{START_SEASON}_{END_SEASON}.parquet",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.out.exists() and not args.force:
        existing = pd.read_parquet(args.out)
        print(f"Already pulled: {len(existing):,} rows in {args.out.name}")
        return

    war_df = pull_war_data(args.start, args.end)
    print()
    rate_df = pull_rate_stats(args.start, args.end)

    if "Name" in rate_df.columns:
        rate_df["Name"] = fix_encoding(rate_df, "Name")
    war_df["name_common"] = fix_encoding(war_df, "name_common")

    print()
    print("Rate stats columns:", rate_df.columns.tolist())
    print()
    print("Merging WAR data with rate stats...")

    # Check what ID column pitching_stats_bref actually provides
    id_col = "mlbID" if "mlbID" in rate_df.columns else None
    if id_col is None:
        print("  WARNING: no mlbID column found in rate_df — inspect columns above")
        merged = rate_df
    else:
        # Coerce both ID columns to a consistent numeric type before merging.
        # rate_df's mlbID sometimes arrives as object dtype (mixed types from
        # HTML scraping); war_df's mlb_ID is float64. errors="coerce" turns
        # any unparseable values into NaN, which simply won't match anything.
        rate_df[id_col] = pd.to_numeric(rate_df[id_col], errors="coerce")
        war_df["mlb_ID"] = pd.to_numeric(war_df["mlb_ID"], errors="coerce")

        n_unparseable = rate_df[id_col].isna().sum()
        if n_unparseable > 0:
            print(f"  Note: {n_unparseable} rate-stat rows had unparseable mlbID, will not match")

        merged = rate_df.merge(
            war_df,
            left_on=[id_col, "year_ID"],
            right_on=["mlb_ID", "year_ID"],
            how="inner",
        )
        print(f"  Merged: {len(merged):,} rows")
        print(f"  Rate stats unmatched: {len(rate_df) - len(merged):,} rows dropped")

    merged.to_parquet(args.out, index=False)
    print(f"\nSaved to {args.out}")

    print()
    print("=== Per-season row counts ===")
    print(merged["year_ID"].value_counts().sort_index())

    print()
    print("=== Columns ===")
    print(sorted(merged.columns.tolist()))

    print()
    print("=== Sample row ===")
    print(merged.iloc[0])


if __name__ == "__main__":
    main()
