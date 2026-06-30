"""Train a logistic regression baseline for pitcher breakout prediction.

Temporal split: train 2015-2021, test 2022-2023. Mirrors the hitter
project's approach with pitcher-specific features.

Usage:
    python -m src.models.train_logreg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

FEATURES = [
    "Age",
    "WAR",
    "GS",
    "IP_decimal",
    "ERA",
    "WHIP",
    "k_per9",
    "bb_per9",
    "hr_per9",
    "k_bb_ratio",
    "war_trend",
    "era_trend",
]

TARGET = "breakout"


def load_data(path: Path):
    df = pd.read_parquet(path)
    df = df[df["has_next_year"] & df["has_prior_year"]].copy()

    train_df = df[df["year_ID"] <= 2021].copy()
    test_df = df[df["year_ID"].isin([2022, 2023])].copy()

    print(f"Train: {len(train_df):,} rows ({train_df[TARGET].mean():.1%} breakout rate)")
    print(f"Test:  {len(test_df):,} rows ({test_df[TARGET].mean():.1%} breakout rate)")

    return train_df, test_df


def evaluate(y_true, y_proba) -> dict:
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=PROCESSED_DIR / "breakout_features.parquet",
    )
    parser.add_argument("--out", type=Path, default=MODELS_DIR / "logreg.joblib")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_data(args.features)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET].astype(int)
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET].astype(int)

    pipeline = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("numeric", StandardScaler(), FEATURES),
        ])),
        ("classifier", LogisticRegressionCV(
            Cs=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
            cv=5,
            scoring="average_precision",
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    print("\nFitting logistic regression (5-fold CV over C)...")
    pipeline.fit(X_train, y_train)

    best_C = pipeline.named_steps["classifier"].C_[0]
    print(f"  Best C selected by 5-fold CV: {best_C:.4f}")

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, y_proba)

    print("\n=== Test metrics (2022-2023) ===")
    print(f"  AUC:          {metrics['auc']:.4f}")
    print(f"  PR-AUC:       {metrics['pr_auc']:.4f}  (baseline = positive rate = {y_test.mean():.4f})")
    print(f"  Brier score:  {metrics['brier_score']:.4f}")
    print(f"  N samples:    {metrics['n_samples']:,} ({metrics['n_positive']} breakouts)")
    print()
    cr = metrics["classification_report"]
    print(f"  Precision (breakout): {cr['1']['precision']:.3f}")
    print(f"  Recall (breakout):    {cr['1']['recall']:.3f}")
    print(f"  F1 (breakout):        {cr['1']['f1-score']:.3f}")

    joblib.dump(pipeline, args.out)
    print(f"\nModel saved to {args.out}")

    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
