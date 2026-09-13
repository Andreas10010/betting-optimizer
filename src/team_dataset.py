"""Lagcentrerad pre-match-data: en rad per lag och match, utan resultatläckage."""
from __future__ import annotations

import numpy as np
import pandas as pd


IDENTIFIER_COLUMNS = [
    "MatchId", "Season", "League", "Team", "MatchDate",
    "Opponent", "Venue", "IsHome", "Target",
]
RESULT_MAP = {
    "home": {"H": "Vinst", "D": "Oavgjort", "A": "Förlust"},
    "away": {"H": "Förlust", "D": "Oavgjort", "A": "Vinst"},
}
HISTORY_METRICS = [
    "points", "goals_for", "goals_against", "goal_diff",
    "shots_for", "shots_against", "sot_for", "sot_against",
    "corners_for", "corners_against", "yellow_cards", "red_cards",
]


def match_id(frame: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(frame["MatchDate"], errors="coerce")
    return (
        frame["League"].astype(str) + "_"
        + dates.dt.strftime("%Y%m%d") + "_"
        + frame["HomeTeam"].astype(str) + "_"
        + frame["AwayTeam"].astype(str)
    )


def add_relative_strength_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Lag-motståndare-skillnader som används av den bästa logistiska modellen."""
    result = frame.copy()
    result["position_advantage"] = result["opponent_position"] - result["team_position"]
    result["elo_difference"] = result["team_elo"] - result["opponent_elo"]
    result["expected_goal_difference"] = result["expected_team_goals"] - result["expected_opponent_goals"]
    return result


def build_team_match_dataset(raw_history, engineered_matches, season_label=None) -> pd.DataFrame:
    """En rad per lag/match med strikt historiska total-, hemma- och bortafeatures."""
    raw_history = raw_history.copy()
    raw_history["MatchDate"] = pd.to_datetime(raw_history["MatchDate"], errors="coerce")
    raw_history["MatchId"] = match_id(raw_history)

    parts = []
    for side, venue in (("home", "Hemma"), ("away", "Borta")):
        own = "Home" if side == "home" else "Away"
        opp = "Away" if side == "home" else "Home"
        part = pd.DataFrame({
            "MatchId": raw_history["MatchId"],
            "Season": raw_history["Season"],
            "League": raw_history["League"],
            "MatchDate": raw_history["MatchDate"],
            "Team": raw_history[f"{own}Team"],
            "Opponent": raw_history[f"{opp}Team"],
            "Venue": venue,
            "IsHome": int(side == "home"),
            "Target": raw_history["FullTimeResult"].map(RESULT_MAP[side]),
            "_points": raw_history["FullTimeResult"].map(
                {"H": 3, "D": 1, "A": 0} if side == "home" else {"H": 0, "D": 1, "A": 3}
            ),
            "_win": (raw_history["FullTimeResult"] == ("H" if side == "home" else "A")).astype(int),
            "_draw": (raw_history["FullTimeResult"] == "D").astype(int),
            "_loss": (raw_history["FullTimeResult"] == ("A" if side == "home" else "H")).astype(int),
            "_goals_for": raw_history[f"FullTime{own}Goals"],
            "_goals_against": raw_history[f"FullTime{opp}Goals"],
            "_shots_for": raw_history[f"{own}Shots"],
            "_shots_against": raw_history[f"{opp}Shots"],
            "_sot_for": raw_history[f"{own}ShotsOnTarget"],
            "_sot_against": raw_history[f"{opp}ShotsOnTarget"],
            "_corners_for": raw_history[f"{own}Corners"],
            "_corners_against": raw_history[f"{opp}Corners"],
            "_yellow_cards": raw_history[f"{own}YellowCards"],
            "_red_cards": raw_history[f"{own}RedCards"],
        })
        part["_goal_diff"] = part["_goals_for"] - part["_goals_against"]
        parts.append(part)

    long = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["Team", "MatchDate", "MatchId"])
        .reset_index(drop=True)
    )

    feature_records = []
    for _, team_matches in long.groupby("Team", sort=False):
        histories = {"total": [], "home": [], "away": []}
        for idx, row in team_matches.iterrows():
            values = {}
            for scope in ("total", "home", "away"):
                prior = histories[scope]
                values[f"history_matches_{scope}"] = len(prior)
                for window in (5, 10):
                    recent = prior[-window:]
                    values[f"matches_{scope}_l{window}"] = len(recent)
                    for metric in HISTORY_METRICS:
                        observed = [item[metric] for item in recent if pd.notna(item[metric])]
                        values[f"{metric}_{scope}_l{window}"] = float(np.sum(observed)) if observed else np.nan
                    values[f"ppg_{scope}_l{window}"] = (
                        values[f"points_{scope}_l{window}"] / len(recent) if recent else np.nan
                    )
                    for outcome in ("win", "draw", "loss"):
                        values[f"{outcome}_rate_{scope}_l{window}"] = (
                            sum(item[outcome] for item in recent) / len(recent) if recent else np.nan
                        )
            feature_records.append((idx, values))

            record = {name: row[f"_{name}"] for name in HISTORY_METRICS}
            record.update(win=row["_win"], draw=row["_draw"], loss=row["_loss"])
            histories["total"].append(record)
            histories["home" if row["IsHome"] else "away"].append(record)

    calculated = pd.DataFrame({idx: values for idx, values in feature_records}).T
    long = pd.concat([long, calculated.reindex(long.index)], axis=1)

    engineered_matches = engineered_matches.copy()
    engineered_matches["MatchDate"] = pd.to_datetime(engineered_matches["MatchDate"], errors="coerce")
    engineered_matches["MatchId"] = match_id(engineered_matches)
    perspectives = []
    for side, opponent in (("home", "away"), ("away", "home")):
        is_home = side == "home"
        perspectives.append(pd.DataFrame({
            "MatchId": engineered_matches["MatchId"],
            "Team": engineered_matches["HomeTeam" if is_home else "AwayTeam"],
            "team_position": engineered_matches[f"{side}_position"],
            "opponent_position": engineered_matches[f"{opponent}_position"],
            "team_season_points": engineered_matches[f"{side}_points"],
            "opponent_season_points": engineered_matches[f"{opponent}_points"],
            "team_season_ppg": engineered_matches[f"{side}_points_per_game"],
            "opponent_season_ppg": engineered_matches[f"{opponent}_points_per_game"],
            "team_elo": engineered_matches[f"{side}_elo"],
            "opponent_elo": engineered_matches[f"{opponent}_elo"],
            "team_elo_expected": engineered_matches[f"{side}_elo_expected"],
            "expected_team_goals": engineered_matches[f"expected_{side}_goals"],
            "expected_opponent_goals": engineered_matches[f"expected_{opponent}_goals"],
            "poisson_prob_win": engineered_matches[f"poisson_prob_{side}"],
            "poisson_prob_draw": engineered_matches["poisson_prob_draw"],
            "poisson_prob_loss": engineered_matches[f"poisson_prob_{opponent}"],
        }))

    perspective = pd.concat(perspectives, ignore_index=True)
    long = long.merge(perspective, on=["MatchId", "Team"], how="left", validate="one_to_one")

    internal = [column for column in long if column.startswith("_")]
    final = long.drop(columns=internal)
    if season_label is not None:
        final = final.loc[final["Season"].eq(season_label)]
    feature_columns = [column for column in final if column not in IDENTIFIER_COLUMNS]
    return (
        add_relative_strength_features(final[IDENTIFIER_COLUMNS + feature_columns])
        .sort_values(["Team", "MatchDate", "MatchId"])
        .reset_index(drop=True)
    )
