"""Build breakout labels and features from raw pitcher season data.

Filters to starting pitchers only (GS >= 15), converts the baseball
innings-pitched notation (X.1 = X innings + 1 out, X.2 = X innings + 2 outs)
to true decimal innings, then builds leakage-free features and a WAR-jump
breakout label exactly mirroring the hitter project's approach.

Usage:
    python -m src.features.build_features
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

WAR_JUMP_THRESHOLD = 2.0
MIN_GS_CURRENT_YEAR = 15   # our "starter" filter, current season
MIN_IP_CURRENT_YEAR = 80   # innings floor, catches the rare short-outing-heavy season
MIN_GS_NEXT_YEAR = 15      # qualification bar for the label year


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.rename(columns={"GS_x": "GS", "G_x": "G"})
    df = df.drop(columns=["GS_y", "G_y", "Name", "mlb_ID"], errors="ignore")
    return df


def fix_innings_notation(ip_raw: pd.Series) -> pd.Series:
    """Convert baseball's X.1/X.2 innings notation to true decimal innings.

    Baseball box scores write 30.2 to mean "30 innings and 2 outs" (i.e.
    30 + 2/3), not 30.2 decimal innings. The fractional part is always
    .0, .1, or .2, representing 0, 1, or 2 outs in the partial inning.
    """
    whole = np.floor(ip_raw)
    frac_digit = np.round((ip_raw - whole) * 10)  # 0, 1, or 2
    return whole + frac_digit / 3.0


def compute_rate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add rate stats normalized by innings/batters faced."""
    df = df.copy()
    ip = df["IP_decimal"].replace(0, np.nan)
    bf = df["BF"].replace(0, np.nan)

    df["k_per9"] = df["SO9"]            # already provided, keep as-is
    df["bb_per9"] = (df["BB"] * 9) / ip
    df["hr_per9"] = (df["HR"] * 9) / ip
    df["k_bb_ratio"] = df["SO/W"]        # already provided

    return df


def build_labeled_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """For each pitcher-season, attach next-year WAR and breakout label."""
    df = df.sort_values(["player_ID", "year_ID"]).reset_index(drop=True)

    next_year = df[["player_ID", "year_ID", "WAR", "GS"]].copy()
    next_year = next_year.rename(
        columns={"year_ID": "year_ID_next", "WAR": "WAR_next", "GS": "GS_next"}
    )
    next_year["year_ID"] = next_year["year_ID_next"] - 1

    merged = df.merge(
        next_year[["player_ID", "year_ID", "WAR_next", "GS_next", "year_ID_next"]],
        on=["player_ID", "year_ID"],
        how="left",
    )

    merged["has_next_year"] = merged["year_ID_next"].notna()
    merged["next_year_qualified"] = merged["GS_next"] >= MIN_GS_NEXT_YEAR
    merged["war_jump"] = merged["WAR_next"] - merged["WAR"]

    merged["breakout"] = (
        merged["has_next_year"]
        & merged["next_year_qualified"]
        & (merged["war_jump"] >= WAR_JUMP_THRESHOLD)
    ).astype(int)

    return merged


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add year-over-year trend features."""
    df = df.sort_values(["player_ID", "year_ID"]).reset_index(drop=True)

    prev_year = df[["player_ID", "year_ID", "WAR", "ERA", "GS"]].copy()
    prev_year = prev_year.rename(
        columns={"WAR": "WAR_prev", "ERA": "ERA_prev", "GS": "GS_prev"}
    )
    prev_year["year_ID"] = prev_year["year_ID"] + 1

    df = df.merge(
        prev_year[["player_ID", "year_ID", "WAR_prev", "ERA_prev", "GS_prev"]],
        on=["player_ID", "year_ID"],
        how="left",
    )

    df["has_prior_year"] = df["WAR_prev"].notna()
    df["war_trend"] = df["WAR"] - df["WAR_prev"]
    df["era_trend"] = df["ERA_prev"] - df["ERA"]  # positive = improvement (lower ERA)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DIR / "pitchers_2015_2024.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "breakout_features.parquet",
    )
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    df = load_raw(args.input)
    print(f"  Loaded {len(df):,} pitcher-seasons")

    print(f"\nFixing innings-pitched notation...")
    df["IP_decimal"] = fix_innings_notation(df["IP"])

    print(f"\nFiltering to starters (GS >= {MIN_GS_CURRENT_YEAR}, IP >= {MIN_IP_CURRENT_YEAR})...")
    df = df[(df["GS"] >= MIN_GS_CURRENT_YEAR) & (df["IP_decimal"] >= MIN_IP_CURRENT_YEAR)].copy()
    print(f"  Remaining: {len(df):,} pitcher-seasons")

    print("\nComputing rate stats...")
    df = compute_rate_stats(df)

    print(f"\nBuilding breakout labels (WAR jump >= {WAR_JUMP_THRESHOLD} next year)...")
    df = build_labeled_dataset(df)

    print("\nAdding year-over-year trend features...")
    df = add_trend_features(df)

    labeled = df[df["has_next_year"]].copy()

    print(f"\n=== Label summary (labeled rows only: {len(labeled):,}) ===")
    print(labeled["breakout"].value_counts())
    print(f"Breakout rate: {labeled['breakout'].mean():.3%}")

    print(f"\n=== Full dataset (including unlabeled final-season rows): {len(df):,} ===")
    df.to_parquet(args.output, index=False)
    print(f"Saved to {args.output}")

    print("\n=== war_jump distribution (for labeled rows) ===")
    print(labeled["war_jump"].describe())

    print("\n=== IP_decimal sanity check (should look like normal decimals now) ===")
    print(df[["IP", "IP_decimal"]].head(10))


if __name__ == "__main__":
    main()
