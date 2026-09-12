"""Reproducible feature-family benchmark: python -m src.evaluate_features."""
from __future__ import annotations

import argparse
from pathlib import Path

from .data_loader import load_raw_data
from .feature_engineering import engineer_features
from .modeling import FEATURE_GROUPS, walk_forward_compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Local CSV; otherwise use the project's public dataset")
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--groups", nargs="+", choices=list(FEATURE_GROUPS), default=list(FEATURE_GROUPS))
    parser.add_argument("--output", type=Path, default=Path("reports/feature_benchmark"))
    args = parser.parse_args()
    raw = load_raw_data(path=args.data, nrows=args.rows)
    print(f"Engineering {len(raw):,} matches...", flush=True)
    features = engineer_features(raw)
    results, calibration = walk_forward_compare(
        features, {name: FEATURE_GROUPS[name] for name in args.groups}, n_splits=args.folds)
    args.output.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output / "fold_metrics.csv", index=False)
    calibration.to_csv(args.output / "calibration.csv", index=False)
    weighted = results.copy()
    metrics = ["log_loss", "brier_score", "accuracy"]
    weighted[metrics] = weighted[metrics].mul(weighted["eval_rows"], axis=0)
    sums = weighted.groupby("feature_group")[[*metrics, "eval_rows"]].sum()
    summary = sums[metrics].div(sums["eval_rows"], axis=0).sort_values("log_loss")
    summary.to_csv(args.output / "summary.csv")
    print(summary.to_string(), flush=True)
    print(f"Final holdout starts {results.holdout_start.iloc[0]}; excluded from this comparison.", flush=True)
    print(f"Saved fold metrics and calibration to {args.output}", flush=True)


if __name__ == "__main__":
    main()
