"""The features must depend only on earlier matches.

A match's features legitimately use results from earlier in the same season:
that information exists when the bet is placed. What must never happen is a
feature depending on a *later* match. Both are hard to tell apart by reading
the code, but they differ in one observable way: deleting the future must not
change the past. These tests delete it and compare.
"""
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import engineer_features
from src.team_dataset import build_team_match_dataset
from src.team_model import numeric_feature_columns

CUT = pd.Timestamp("2020-12-01")  # Mid-way through the fixture list, not after it.


def synthetic_matches() -> pd.DataFrame:
    """Two seasons of a six-team league, one round of fixtures per week."""
    teams = [f"Team {name}" for name in "ABCDEFGH"]
    rng = np.random.default_rng(7)
    rows, date = [], pd.Timestamp("2020-08-01")
    for season, (start, end) in {"2020/21": (0, 15), "2021/22": (15, 30)}.items():
        for week in range(start, end):
            shuffled = list(rng.permutation(teams))
            for home, away in zip(shuffled[::2], shuffled[1::2]):
                home_goals, away_goals = int(rng.integers(0, 4)), int(rng.integers(0, 4))
                rows.append({
                    "Season": season, "League": "Test League", "MatchDate": date,
                    "HomeTeam": home, "AwayTeam": away,
                    "FullTimeHomeGoals": home_goals, "FullTimeAwayGoals": away_goals,
                    "FullTimeResult": "H" if home_goals > away_goals else "D" if home_goals == away_goals else "A",
                    "HalfTimeHomeGoals": 0, "HalfTimeAwayGoals": 0, "HalfTimeResult": "D",
                    "HomeShots": 12, "AwayShots": 10, "HomeShotsOnTarget": 5, "AwayShotsOnTarget": 4,
                    "HomeCorners": 5, "AwayCorners": 4, "HomeFouls": 11, "AwayFouls": 12,
                    "HomeYellowCards": 1, "AwayYellowCards": 2, "HomeRedCards": 0, "AwayRedCards": 0,
                })
            date += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


def _columns_that_change_when_the_future_is_deleted(build, columns=None):
    matches = synthetic_matches()
    full = build(matches)
    truncated = build(matches.loc[matches["MatchDate"] < CUT].copy())
    past = full.loc[full["MatchDate"] < CUT]
    shared = past.index.intersection(truncated.index)
    assert len(shared) > 60, "truncation must remove some matches but keep many"
    past, truncated = past.loc[shared], truncated.loc[shared]
    candidates = columns or [c for c in past.columns
                             if past[c].dtype.kind in "fi" and c in truncated.columns]
    return [name for name in candidates
            if not np.allclose(past[name].to_numpy(float), truncated[name].to_numpy(float),
                               rtol=1e-9, atol=1e-9, equal_nan=True)]


def test_engineered_features_do_not_change_when_later_matches_are_deleted():
    def build(matches):
        engineered = engineer_features(matches)
        return engineered.set_index(
            engineered["League"] + "|" + engineered["MatchDate"].astype(str)
            + "|" + engineered["HomeTeam"] + "|" + engineered["AwayTeam"]).sort_index()

    leaked = _columns_that_change_when_the_future_is_deleted(build)
    assert leaked == [], f"these features can see the future: {leaked}"


def test_the_columns_the_model_consumes_do_not_change_either():
    def build(matches):
        team_data = build_team_match_dataset(matches, engineer_features(matches))
        return team_data.set_index(team_data["MatchId"] + "|" + team_data["Team"]).sort_index()

    leaked = _columns_that_change_when_the_future_is_deleted(build, numeric_feature_columns())
    assert leaked == [], f"these model inputs can see the future: {leaked}"


def test_the_check_would_actually_catch_a_leak():
    """A guard is worthless if it cannot fail. Plant a leak and confirm it trips."""
    def build(matches):
        engineered = engineer_features(matches)
        frame = engineered.set_index(
            engineered["League"] + "|" + engineered["MatchDate"].astype(str)
            + "|" + engineered["HomeTeam"] + "|" + engineered["AwayTeam"]).sort_index()
        # A season-wide mean: every row sees every other row, including later ones.
        frame["planted_leak"] = frame.groupby("Season")["FullTimeHomeGoals"].transform("mean")
        return frame

    assert _columns_that_change_when_the_future_is_deleted(build) == ["planted_leak"]
