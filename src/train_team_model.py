"""Train the all-history logistic model and save it for the dashboard.

Run from the repository root:

    PYTHONPATH=src python src/train_team_model.py
"""
from __future__ import annotations

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_loader import load_raw_data
from feature_engineering import engineer_features
from team_dataset import build_team_match_dataset
from team_model import (
    LEAGUE,
    TEST_SEASON,
    artifact_path,
    save_trained_model,
    train_all_history_model,
)


def build_team_data(through_season=TEST_SEASON):
    raw = load_raw_data()
    history = raw.loc[raw["League"].eq(LEAGUE) & raw["Season"].astype(str).le(through_season)].copy()
    if history.empty:
        raise ValueError(f"Ingen data för {LEAGUE} t.o.m. {through_season}.")
    engineered = engineer_features(history)
    return build_team_match_dataset(history, engineered, season_label=None)


def main():
    path = artifact_path()
    print(f"Bygger Premier League-features t.o.m. {TEST_SEASON}...")
    team_data = build_team_data()
    print(f"Tränar LG på {team_data.loc[team_data.Season.astype(str).lt(TEST_SEASON)].shape[0]:,} lagrader...")
    bundle = train_all_history_model(team_data)
    save_trained_model(path, bundle)
    metrics = bundle["metrics"]
    print(
        f"Sparad: {path}\n"
        f"Train: {bundle['train_rows']:,} rader, {bundle['train_start']}–{bundle['train_end']}\n"
        f"Test {TEST_SEASON}: accuracy {metrics['accuracy']:.3f}, log loss {metrics['log_loss']:.3f}"
    )


if __name__ == "__main__":
    main()
