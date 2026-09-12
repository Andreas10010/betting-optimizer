from __future__ import annotations

from collections import defaultdict

from scipy.stats import skellam

try:  # Support both Streamlit scripts and package imports.
    from .count_ratings import CountRatings
except ImportError:
    from count_ratings import CountRatings

import numpy as np
import pandas as pd


NUMERIC_MATCH_COLUMNS = [
    "FullTimeHomeGoals",
    "FullTimeAwayGoals",
    "HalfTimeHomeGoals",
    "HalfTimeAwayGoals",
    "HomeShots",
    "AwayShots",
    "HomeShotsOnTarget",
    "AwayShotsOnTarget",
    "HomeCorners",
    "AwayCorners",
    "HomeFouls",
    "AwayFouls",
    "HomeYellowCards",
    "AwayYellowCards",
    "HomeRedCards",
    "AwayRedCards",
]

FEATURE_COLUMNS = [
    "home_form_points_last_5",
    "away_form_points_last_5",
    "home_goals_last_5",
    "away_goals_last_5",
    "home_goals_against_last_5",
    "away_goals_against_last_5",
    "home_shots_last_5",
    "away_shots_last_5",
    "home_shots_on_target_last_5",
    "away_shots_on_target_last_5",
    "home_corners_last_5",
    "away_corners_last_5",
    "home_yellow_cards_last_5",
    "away_yellow_cards_last_5",
    "home_red_cards_last_5",
    "away_red_cards_last_5",
]

FORM_FEATURE_COLUMNS = [
    "home_recent_form_5",
    "away_recent_form_5",
    "home_recent_form_8",
    "away_recent_form_8",
    "home_recent_form_5_avg",
    "away_recent_form_5_avg",
    "home_recent_form_8_avg",
    "away_recent_form_8_avg",
    "home_recent_goal_diff_5",
    "away_recent_goal_diff_5",
    "home_recent_goal_diff_8",
    "away_recent_goal_diff_8",
]

EXTRA_FEATURE_COLUMNS = [
    "home_points_last_5",
    "away_points_last_5",
    "home_points_last_10",
    "away_points_last_10",
    "home_ppg_l5",
    "away_ppg_l5",
    "home_ppg_l10",
    "away_ppg_l10",
    "home_ppg_momentum_l3_l10",
    "away_ppg_momentum_l3_l10",
    "home_win_rate_l5",
    "away_win_rate_l5",
    "home_draw_rate_l5",
    "away_draw_rate_l5",
    "home_loss_rate_l5",
    "away_loss_rate_l5",
    "home_goals_for_l5",
    "away_goals_for_l5",
    "home_goals_against_l5",
    "away_goals_against_l5",
    "home_goals_for_l10",
    "away_goals_for_l10",
    "home_goals_against_l10",
    "away_goals_against_l10",
    "home_goal_diff_l5",
    "away_goal_diff_l5",
    "home_goal_diff_l10",
    "away_goal_diff_l10",
    "home_shots_for_l5",
    "away_shots_for_l5",
    "home_shots_against_l5",
    "away_shots_against_l5",
    "home_shots_for_l10",
    "away_shots_for_l10",
    "home_shots_against_l10",
    "away_shots_against_l10",
    "home_sot_for_l5",
    "away_sot_for_l5",
    "home_sot_against_l5",
    "away_sot_against_l5",
    "home_sot_for_l10",
    "away_sot_for_l10",
    "home_sot_against_l10",
    "away_sot_against_l10",
    "home_corners_for_l5",
    "away_corners_for_l5",
    "home_corners_against_l5",
    "away_corners_against_l5",
    "home_corners_for_l10",
    "away_corners_for_l10",
    "home_corners_against_l10",
    "away_corners_against_l10",
    "home_home_ppg_l5",
    "away_away_ppg_l5",
    "home_home_ppg_l10",
    "away_away_ppg_l10",
    "home_home_goals_for_l5",
    "away_away_goals_for_l5",
    "home_home_goals_against_l5",
    "away_away_goals_against_l5",
    "home_home_shots_for_l5",
    "away_away_shots_for_l5",
    "home_home_sot_for_l5",
    "away_away_sot_for_l5",
    "home_home_corners_for_l5",
    "away_away_corners_for_l5",
    "home_shot_accuracy_l5",
    "away_shot_accuracy_l5",
    "home_sot_conversion_l5",
    "away_sot_conversion_l5",
    "home_attack_strength_l10",
    "away_attack_strength_l10",
    "home_defensive_strength_l10",
    "away_defensive_strength_l10",
    "home_position",
    "away_position",
    "position_diff",
    "home_points",
    "away_points",
    "home_points_per_game",
    "away_points_per_game",
    "points_per_game_diff",
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_elo_expected",
    "away_elo_expected",
    "home_sot_momentum_l3_l10",
    "away_sot_momentum_l3_l10",
    "home_shot_momentum_l3_l10",
    "away_shot_momentum_l3_l10",
    "home_corner_momentum_l3_l10",
    "away_corner_momentum_l3_l10",
]


# Compatibility aliases are exported, but never selected twice by the model.
FEATURE_ALIASES = {}
for _side in ("home", "away"):
    for _alias, _canonical in {
        "form_points_last_5": "points_last_5", "goals_last_5": "goals_for_l5",
        "goals_against_last_5": "goals_against_l5", "shots_last_5": "shots_for_l5",
        "shots_on_target_last_5": "sot_for_l5", "corners_last_5": "corners_for_l5",
        "recent_goal_diff_5": "goal_diff_l5",
    }.items():
        FEATURE_ALIASES[f"{_side}_{_alias}"] = f"{_side}_{_canonical}"

CONTEXT_FEATURE_COLUMNS = [f"{side}_{name}" for side in ("home", "away") for name in (
    "history_matches", "history_matches_l10", "shot_observations_l10",
    "rest_days", "matches_last_14_days", "shot_share_l10", "sot_share_l10",
    "sot_diff_per_match_l10", "opponent_elo_mean_l5", "result_overperformance_l5",
    "goal_diff_std_l10", "goals_for_ewm", "goals_against_ewm", "shots_for_ewm",
    "shots_against_ewm", "sot_for_ewm", "sot_against_ewm",
)] + ["home_adjusted_elo_diff", "rest_days_diff"]
GOAL_MODEL_FEATURE_COLUMNS = [
    "expected_home_goals", "expected_away_goals", "expected_total_goals", "expected_goal_diff",
    "poisson_prob_home", "poisson_prob_draw", "poisson_prob_away",
]
SHOT_RATING_FEATURE_COLUMNS = [
    f"{side}_{stat}_{kind}_rating" for side in ("home", "away")
    for stat in ("shot", "sot") for kind in ("attack", "defence")
] + ["expected_home_shots", "expected_away_shots", "expected_home_sot", "expected_away_sot"]
NEW_FEATURE_COLUMNS = CONTEXT_FEATURE_COLUMNS + GOAL_MODEL_FEATURE_COLUMNS + SHOT_RATING_FEATURE_COLUMNS
DERIVED_FEATURE_COLUMNS = ["team_strength_delta", "goal_delta_last_5", "shot_delta_last_5",
                           "recent_form_delta_5", "recent_form_delta_8"]
ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + FORM_FEATURE_COLUMNS + EXTRA_FEATURE_COLUMNS + NEW_FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if np.isfinite(denominator) and denominator > 0 else np.nan


def _make_team_entry(row, team_is_home):
    own, other = ("Home", "Away") if team_is_home else ("Away", "Home")
    gf, ga = row[f"FullTime{own}Goals"], row[f"FullTime{other}Goals"]
    result = int(np.sign(gf - ga))
    entry = {
        "points": {1: 3, 0: 1, -1: 0}[result], "form_score": result,
        "goal_diff": gf - ga, "goals_for": gf, "goals_against": ga,
        "is_home": int(team_is_home), "date": row["MatchDate"],
    }
    for metric, column in {"shots": "Shots", "shots_on_target": "ShotsOnTarget",
                           "corners": "Corners", "yellow_cards": "YellowCards",
                           "red_cards": "RedCards", "fouls": "Fouls"}.items():
        entry[metric] = row[f"{own}{column}"]
    for metric, column in {"shots_against": "Shots", "shots_on_target_against": "ShotsOnTarget",
                           "corners_against": "Corners"}.items():
        entry[metric] = row[f"{other}{column}"]
    return entry


HISTORY_METRICS = ("points", "form_score", "goal_diff", "goals_for", "goals_against",
                   "shots", "shots_on_target", "corners", "yellow_cards", "red_cards", "fouls",
                   "shots_against", "shots_on_target_against", "corners_against")


def _summarize_history(history, window=5):
    recent = history[-window:]
    summary = {"matches": len(recent)}
    for metric in HISTORY_METRICS:
        values = np.asarray([entry[metric] for entry in recent], dtype=float)
        valid = values[np.isfinite(values)]
        # Totals require complete windows; means use observed matches only.
        summary[metric] = float(valid.sum()) if len(valid) == len(recent) and len(valid) else np.nan
        summary[f"{metric}_mean"] = float(valid.mean()) if len(valid) else np.nan
        summary[f"{metric}_count"] = len(valid)
    summary["form_avg"] = summary["form_score_mean"]
    summary["ppg"] = summary["points_mean"]
    for name, score in (("wins", 1), ("draws", 0), ("losses", -1)):
        summary[name] = float(sum(entry["form_score"] == score for entry in recent)) if recent else np.nan
    return summary


def _build_standings(league_stats):
    ranking = sorted(league_stats.items(), key=lambda item: (
        -item[1]["points"], -item[1]["goal_diff"], -item[1]["goals_for"], item[0]))
    return {team: i for i, (team, _) in enumerate(ranking, 1)}


def _context_features(history, date):
    recent = history[-10:]
    dated_history = []
    for entry in reversed(history):
        age = (date - entry["date"]).days
        if age > 1095:
            break
        dated_history.append((entry, age))
    features = {
        "history_matches": len(history), "history_matches_l10": len(recent),
        "shot_observations_l10": sum(np.isfinite(x["shots"]) for x in recent),
        "rest_days": (date - history[-1]["date"]).days if history else np.nan,
        "matches_last_14_days": sum(age <= 14 for entry, age in dated_history),
        "opponent_elo_mean_l5": np.mean([x["opponent_elo"] for x in history[-5:]]) if history else np.nan,
        "result_overperformance_l5": np.mean([x["overperformance"] for x in history[-5:]]) if history else np.nan,
        "goal_diff_std_l10": np.std([x["goal_diff"] for x in recent], ddof=1) if len(recent) >= 2 else np.nan,
    }
    for name, own, against in (("shot", "shots", "shots_against"),
                               ("sot", "shots_on_target", "shots_on_target_against")):
        pairs = [(x[own], x[against]) for x in recent if np.isfinite(x[own]) and np.isfinite(x[against])]
        features[f"{name}_share_l10"] = _safe_ratio(sum(a for a, b in pairs), sum(a + b for a, b in pairs))
        if name == "sot":
            features["sot_diff_per_match_l10"] = np.mean([a - b for a, b in pairs]) if pairs else np.nan
    for name, metric in {"goals_for": "goals_for", "goals_against": "goals_against",
                         "shots_for": "shots", "shots_against": "shots_against",
                         "sot_for": "shots_on_target", "sot_against": "shots_on_target_against"}.items():
        # Finite 3-year window; 60 calendar days is a fixed, tunable half-life.
        observations = [(entry[metric], age) for entry, age in dated_history if np.isfinite(entry[metric])]
        features[f"{name}_ewm"] = (float(np.average([v for v, age in observations],
            weights=[2 ** (-age / 60) for v, age in observations])) if observations else np.nan)
    return features


def _prepare_matches(df, season_start_month):
    if not 1 <= season_start_month <= 12:
        raise ValueError("season_start_month must be between 1 and 12")
    out = df.copy()
    required = ["MatchDate", "League", "HomeTeam", "AwayTeam", "FullTimeHomeGoals", "FullTimeAwayGoals"]
    missing = set(required) - set(out.columns)
    if missing:
        raise ValueError(f"Missing match columns: {sorted(missing)}")
    out["MatchDate"] = pd.to_datetime(out["MatchDate"], errors="coerce").dt.normalize()
    if out[["MatchDate", "League", "HomeTeam", "AwayTeam"]].isna().any().any():
        raise ValueError("Every match needs a valid date, league and team names")
    if (out["HomeTeam"] == out["AwayTeam"]).any():
        raise ValueError("A team cannot play itself")
    if "Season" not in out:
        year = out["MatchDate"].dt.year - (out["MatchDate"].dt.month < season_start_month).astype(int)
        out["Season"] = year.astype(str)
    elif out["Season"].isna().any():
        raise ValueError("Season must be populated when provided")
    for col in NUMERIC_MATCH_COLUMNS:
        values = pd.to_numeric(out[col], errors="coerce") if col in out else pd.Series(np.nan, index=out.index)
        out[col] = values.where(np.isfinite(values) & (values >= 0) & (values % 1 == 0))
    if "FullTimeResult" not in out:
        out["FullTimeResult"] = pd.Series(pd.NA, index=out.index, dtype="object")
    played = out[["FullTimeHomeGoals", "FullTimeAwayGoals"]].notna().all(axis=1)
    inferred = pd.Series(np.where(out["FullTimeHomeGoals"] > out["FullTimeAwayGoals"], "H",
                         np.where(out["FullTimeHomeGoals"] == out["FullTimeAwayGoals"], "D", "A")), index=out.index)
    inconsistent = played & out["FullTimeResult"].notna() & out["FullTimeResult"].ne(inferred)
    if inconsistent.any():
        raise ValueError("FullTimeResult disagrees with the full-time goals")
    out["target"] = inferred.where(played)
    out["target_label"] = out["target"].map({"H": 0, "D": 1, "A": 2})
    # Duplicate fixtures would count results twice. Repeated teams on one date
    # also make rest/form ordering ambiguous in this date-only dataset.
    participants = pd.concat([out[["League", "MatchDate", team]].rename(columns={team: "Team"})
                              for team in ("HomeTeam", "AwayTeam")], ignore_index=True)
    if participants.duplicated(["League", "MatchDate", "Team"]).any():
        raise ValueError("Duplicate fixture or team playing twice on one date")
    return out.sort_values(["MatchDate", "League", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def engineer_features(df: pd.DataFrame, *, season_start_month: int = 7,
                      elo_home_advantage: float = 60.0) -> pd.DataFrame:
    """Build features from completed matches on strictly earlier calendar dates.

    Explicit Season labels win over the July fallback for European leagues.
    Form and Elo carry across seasons within each league; standings reset.
    """
    if not np.isfinite(elo_home_advantage):
        raise ValueError("elo_home_advantage must be finite")
    out = _prepare_matches(df, season_start_month)
    team_history, team_home_history, team_away_history = defaultdict(list), defaultdict(list), defaultdict(list)
    league_team_stats = defaultdict(dict)
    league_goals_total, league_matches_total = defaultdict(float), defaultdict(int)
    team_elo = defaultdict(lambda: 1500.0)
    count_models = defaultdict(lambda: {"goals": CountRatings(1.2), "shots": CountRatings(10.0), "sot": CountRatings(3.5)})
    records = []
    empty_stats = {"points": 0, "goal_diff": 0, "goals_for": 0, "goals_against": 0, "played": 0}

    for date, day in out.groupby("MatchDate", sort=True):
        pending = []
        for _, row in day.iterrows():
            league, home_team, away_team = row["League"], row["HomeTeam"], row["AwayTeam"]
            season_key = (league, row["Season"])
            home_key, away_key = (league, home_team), (league, away_team)
            league_stats = league_team_stats[season_key]
            # Only participants known by this date; no future season roster.
            standings = _build_standings({home_team: empty_stats, away_team: empty_stats, **league_stats})
            home_elo, away_elo = float(team_elo[home_key]), float(team_elo[away_key])
            adjusted_elo_diff = home_elo + elo_home_advantage - away_elo
            home_expected = float(1 / (1 + 10 ** (-adjusted_elo_diff / 400)))
            away_expected = 1 - home_expected
            home_history, away_history = team_history[home_key], team_history[away_key]
            home_home_history, away_away_history = team_home_history[home_key], team_away_history[away_key]
            home_metrics_3, away_metrics_3 = _summarize_history(home_history, 3), _summarize_history(away_history, 3)
            home_metrics_5, away_metrics_5 = _summarize_history(home_history, 5), _summarize_history(away_history, 5)
            home_metrics_8, away_metrics_8 = _summarize_history(home_history, 8), _summarize_history(away_history, 8)
            home_metrics_10, away_metrics_10 = _summarize_history(home_history, 10), _summarize_history(away_history, 10)
            home_home_metrics_5, away_away_metrics_5 = _summarize_history(home_home_history, 5), _summarize_history(away_away_history, 5)
            home_home_metrics_10, away_away_metrics_10 = _summarize_history(home_home_history, 10), _summarize_history(away_away_history, 10)
            # Team goals per game, rather than total goals by both teams.
            league_avg_goals_per_match = _safe_ratio(league_goals_total[season_key], 2 * league_matches_total[season_key])
            features = {}
            features["home_form_points_last_5"] = home_metrics_5["points"]
            features["away_form_points_last_5"] = away_metrics_5["points"]
            features["home_goals_last_5"] = home_metrics_5["goals_for"]
            features["away_goals_last_5"] = away_metrics_5["goals_for"]
            features["home_goals_against_last_5"] = home_metrics_5["goals_against"]
            features["away_goals_against_last_5"] = away_metrics_5["goals_against"]
            features["home_shots_last_5"] = home_metrics_5["shots"]
            features["away_shots_last_5"] = away_metrics_5["shots"]
            features["home_shots_on_target_last_5"] = home_metrics_5["shots_on_target"]
            features["away_shots_on_target_last_5"] = away_metrics_5["shots_on_target"]
            features["home_corners_last_5"] = home_metrics_5["corners"]
            features["away_corners_last_5"] = away_metrics_5["corners"]
            features["home_yellow_cards_last_5"] = home_metrics_5["yellow_cards"]
            features["away_yellow_cards_last_5"] = away_metrics_5["yellow_cards"]
            features["home_red_cards_last_5"] = home_metrics_5["red_cards"]
            features["away_red_cards_last_5"] = away_metrics_5["red_cards"]

            features["home_recent_form_5"] = home_metrics_5["form_score"]
            features["away_recent_form_5"] = away_metrics_5["form_score"]
            features["home_recent_form_8"] = home_metrics_8["form_score"]
            features["away_recent_form_8"] = away_metrics_8["form_score"]
            features["home_recent_form_5_avg"] = home_metrics_5["form_avg"]
            features["away_recent_form_5_avg"] = away_metrics_5["form_avg"]
            features["home_recent_form_8_avg"] = home_metrics_8["form_avg"]
            features["away_recent_form_8_avg"] = away_metrics_8["form_avg"]
            features["home_recent_goal_diff_5"] = home_metrics_5["goal_diff"]
            features["away_recent_goal_diff_5"] = away_metrics_5["goal_diff"]
            features["home_recent_goal_diff_8"] = home_metrics_8["goal_diff"]
            features["away_recent_goal_diff_8"] = away_metrics_8["goal_diff"]

            features["home_points_last_5"] = home_metrics_5["points"]
            features["away_points_last_5"] = away_metrics_5["points"]
            features["home_points_last_10"] = home_metrics_10["points"]
            features["away_points_last_10"] = away_metrics_10["points"]
            features["home_ppg_l5"] = home_metrics_5["ppg"]
            features["away_ppg_l5"] = away_metrics_5["ppg"]
            features["home_ppg_l10"] = home_metrics_10["ppg"]
            features["away_ppg_l10"] = away_metrics_10["ppg"]
            features["home_ppg_momentum_l3_l10"] = home_metrics_3["ppg"] - home_metrics_10["ppg"]
            features["away_ppg_momentum_l3_l10"] = away_metrics_3["ppg"] - away_metrics_10["ppg"]
            features["home_win_rate_l5"] = home_metrics_5["wins"] / home_metrics_5["matches"] if home_metrics_5["matches"] else np.nan
            features["away_win_rate_l5"] = away_metrics_5["wins"] / away_metrics_5["matches"] if away_metrics_5["matches"] else np.nan
            features["home_draw_rate_l5"] = home_metrics_5["draws"] / home_metrics_5["matches"] if home_metrics_5["matches"] else np.nan
            features["away_draw_rate_l5"] = away_metrics_5["draws"] / away_metrics_5["matches"] if away_metrics_5["matches"] else np.nan
            features["home_loss_rate_l5"] = home_metrics_5["losses"] / home_metrics_5["matches"] if home_metrics_5["matches"] else np.nan
            features["away_loss_rate_l5"] = away_metrics_5["losses"] / away_metrics_5["matches"] if away_metrics_5["matches"] else np.nan

            features["home_goals_for_l5"] = home_metrics_5["goals_for"]
            features["away_goals_for_l5"] = away_metrics_5["goals_for"]
            features["home_goals_against_l5"] = home_metrics_5["goals_against"]
            features["away_goals_against_l5"] = away_metrics_5["goals_against"]
            features["home_goals_for_l10"] = home_metrics_10["goals_for"]
            features["away_goals_for_l10"] = away_metrics_10["goals_for"]
            features["home_goals_against_l10"] = home_metrics_10["goals_against"]
            features["away_goals_against_l10"] = away_metrics_10["goals_against"]
            features["home_goal_diff_l5"] = home_metrics_5["goal_diff"]
            features["away_goal_diff_l5"] = away_metrics_5["goal_diff"]
            features["home_goal_diff_l10"] = home_metrics_10["goal_diff"]
            features["away_goal_diff_l10"] = away_metrics_10["goal_diff"]

            features["home_shots_for_l5"] = home_metrics_5["shots"]
            features["away_shots_for_l5"] = away_metrics_5["shots"]
            features["home_shots_against_l5"] = home_metrics_5["shots_against"]
            features["away_shots_against_l5"] = away_metrics_5["shots_against"]
            features["home_shots_for_l10"] = home_metrics_10["shots"]
            features["away_shots_for_l10"] = away_metrics_10["shots"]
            features["home_shots_against_l10"] = home_metrics_10["shots_against"]
            features["away_shots_against_l10"] = away_metrics_10["shots_against"]

            features["home_sot_for_l5"] = home_metrics_5["shots_on_target"]
            features["away_sot_for_l5"] = away_metrics_5["shots_on_target"]
            features["home_sot_against_l5"] = home_metrics_5["shots_on_target_against"]
            features["away_sot_against_l5"] = away_metrics_5["shots_on_target_against"]
            features["home_sot_for_l10"] = home_metrics_10["shots_on_target"]
            features["away_sot_for_l10"] = away_metrics_10["shots_on_target"]
            features["home_sot_against_l10"] = home_metrics_10["shots_on_target_against"]
            features["away_sot_against_l10"] = away_metrics_10["shots_on_target_against"]

            features["home_corners_for_l5"] = home_metrics_5["corners"]
            features["away_corners_for_l5"] = away_metrics_5["corners"]
            features["home_corners_against_l5"] = home_metrics_5["corners_against"]
            features["away_corners_against_l5"] = away_metrics_5["corners_against"]
            features["home_corners_for_l10"] = home_metrics_10["corners"]
            features["away_corners_for_l10"] = away_metrics_10["corners"]
            features["home_corners_against_l10"] = home_metrics_10["corners_against"]
            features["away_corners_against_l10"] = away_metrics_10["corners_against"]

            features["home_home_ppg_l5"] = home_home_metrics_5["ppg"]
            features["away_away_ppg_l5"] = away_away_metrics_5["ppg"]
            features["home_home_ppg_l10"] = home_home_metrics_10["ppg"]
            features["away_away_ppg_l10"] = away_away_metrics_10["ppg"]
            features["home_home_goals_for_l5"] = home_home_metrics_5["goals_for"]
            features["away_away_goals_for_l5"] = away_away_metrics_5["goals_for"]
            features["home_home_goals_against_l5"] = home_home_metrics_5["goals_against"]
            features["away_away_goals_against_l5"] = away_away_metrics_5["goals_against"]
            features["home_home_shots_for_l5"] = home_home_metrics_5["shots"]
            features["away_away_shots_for_l5"] = away_away_metrics_5["shots"]
            features["home_home_sot_for_l5"] = home_home_metrics_5["shots_on_target"]
            features["away_away_sot_for_l5"] = away_away_metrics_5["shots_on_target"]
            features["home_home_corners_for_l5"] = home_home_metrics_5["corners"]
            features["away_away_corners_for_l5"] = away_away_metrics_5["corners"]

            features["home_shot_accuracy_l5"] = home_metrics_5["shots_on_target"] / home_metrics_5["shots"] if home_metrics_5["shots"] else np.nan
            features["away_shot_accuracy_l5"] = away_metrics_5["shots_on_target"] / away_metrics_5["shots"] if away_metrics_5["shots"] else np.nan
            features["home_sot_conversion_l5"] = home_metrics_5["goals_for"] / home_metrics_5["shots_on_target"] if home_metrics_5["shots_on_target"] else np.nan
            features["away_sot_conversion_l5"] = away_metrics_5["goals_for"] / away_metrics_5["shots_on_target"] if away_metrics_5["shots_on_target"] else np.nan

            home_attack_strength = _safe_ratio(home_metrics_10["goals_for_mean"], league_avg_goals_per_match)
            away_attack_strength = _safe_ratio(away_metrics_10["goals_for_mean"], league_avg_goals_per_match)
            home_defensive_strength = _safe_ratio(home_metrics_10["goals_against_mean"], league_avg_goals_per_match)
            away_defensive_strength = _safe_ratio(away_metrics_10["goals_against_mean"], league_avg_goals_per_match)

            features["home_attack_strength_l10"] = home_attack_strength
            features["away_attack_strength_l10"] = away_attack_strength
            features["home_defensive_strength_l10"] = home_defensive_strength
            features["away_defensive_strength_l10"] = away_defensive_strength

            home_position = standings.get(home_team)
            away_position = standings.get(away_team)
            features["home_position"] = home_position
            features["away_position"] = away_position
            if home_position is not None and away_position is not None:
                features["position_diff"] = home_position - away_position
            else:
                features["position_diff"] = np.nan

            home_stats = league_stats.get(home_team)
            away_stats = league_stats.get(away_team)
            features["home_points"] = home_stats["points"] if home_stats else 0
            features["away_points"] = away_stats["points"] if away_stats else 0
            features["home_points_per_game"] = home_stats["points"] / home_stats["played"] if home_stats and home_stats["played"] else np.nan
            features["away_points_per_game"] = away_stats["points"] / away_stats["played"] if away_stats and away_stats["played"] else np.nan
            features["points_per_game_diff"] = features["home_points_per_game"] - features["away_points_per_game"]

            features["home_elo"] = home_elo
            features["away_elo"] = away_elo
            features["elo_diff"] = home_elo - away_elo
            features["home_elo_expected"] = home_expected
            features["away_elo_expected"] = away_expected

            features["home_sot_momentum_l3_l10"] = home_metrics_3["shots_on_target_mean"] - home_metrics_10["shots_on_target_mean"]
            features["away_sot_momentum_l3_l10"] = away_metrics_3["shots_on_target_mean"] - away_metrics_10["shots_on_target_mean"]
            features["home_shot_momentum_l3_l10"] = home_metrics_3["shots_mean"] - home_metrics_10["shots_mean"]
            features["away_shot_momentum_l3_l10"] = away_metrics_3["shots_mean"] - away_metrics_10["shots_mean"]
            features["home_corner_momentum_l3_l10"] = home_metrics_3["corners_mean"] - home_metrics_10["corners_mean"]
            features["away_corner_momentum_l3_l10"] = away_metrics_3["corners_mean"] - away_metrics_10["corners_mean"]

            for side, history in (("home", home_history), ("away", away_history)):
                features.update({f"{side}_{name}": value for name, value in _context_features(history, date).items()})
            features["home_adjusted_elo_diff"] = adjusted_elo_diff
            features["rest_days_diff"] = features["home_rest_days"] - features["away_rest_days"]
            models = count_models[league]
            for model in models.values():
                model.fit_before(date)
            hg, ag = models["goals"].predict(home_team, away_team)
            features.update(expected_home_goals=hg, expected_away_goals=ag,
                            expected_total_goals=hg + ag, expected_goal_diff=hg - ag,
                            poisson_prob_home=float(skellam.sf(0, hg, ag)),
                            poisson_prob_draw=float(skellam.pmf(0, hg, ag)),
                            poisson_prob_away=float(skellam.cdf(-1, hg, ag)))
            for stat, name in (("shots", "shot"), ("sot", "sot")):
                expected_home, expected_away = models[stat].predict(home_team, away_team)
                features[f"expected_home_{stat}"] = expected_home
                features[f"expected_away_{stat}"] = expected_away
                for side, team in (("home", home_team), ("away", away_team)):
                    attack, defence = models[stat].strengths(team)
                    features[f"{side}_{name}_attack_rating"] = attack
                    features[f"{side}_{name}_defence_rating"] = defence
            records.append(features)
            if pd.notna(row["target_label"]):
                pending.append((row, home_expected, home_elo, away_elo))

        # Update only once every prediction for the date has been emitted.
        for row, home_expected, home_elo, away_elo in pending:
            league, home, away = row["League"], row["HomeTeam"], row["AwayTeam"]
            season_key = (league, row["Season"])
            home_key, away_key = (league, home), (league, away)
            home_entry, away_entry = _make_team_entry(row, True), _make_team_entry(row, False)
            score = (home_entry["form_score"] + 1) / 2  # W=1, D=.5, L=0
            home_entry.update(opponent_elo=away_elo, overperformance=score - home_expected)
            away_entry.update(opponent_elo=home_elo, overperformance=(1 - score) - (1 - home_expected))
            team_history[home_key].append(home_entry)
            team_history[away_key].append(away_entry)
            team_home_history[home_key].append(home_entry)
            team_away_history[away_key].append(away_entry)
            for team, entry in ((home, home_entry), (away, away_entry)):
                stats = league_team_stats[season_key].setdefault(team, empty_stats.copy())
                for metric in ("points", "goal_diff", "goals_for", "goals_against"):
                    stats[metric] += entry[metric]
                stats["played"] += 1
            league_goals_total[season_key] += home_entry["goals_for"] + away_entry["goals_for"]
            league_matches_total[season_key] += 1
            change = 32 * (score - home_expected)
            team_elo[home_key] += change
            team_elo[away_key] -= change
            for stat, column in (("goals", "FullTime{}Goals"), ("shots", "{}Shots"), ("sot", "{}ShotsOnTarget")):
                count_models[league][stat].observe(date, home, away, row[column.format("Home")], row[column.format("Away")])

    feature_df = pd.DataFrame(records, index=out.index).reindex(columns=ALL_FEATURE_COLUMNS)
    for name, metric in {"team_strength_delta": "points_last_5", "goal_delta_last_5": "goals_for_l5",
                         "shot_delta_last_5": "shots_for_l5", "recent_form_delta_5": "recent_form_5",
                         "recent_form_delta_8": "recent_form_8"}.items():
        feature_df[name] = feature_df[f"home_{metric}"] - feature_df[f"away_{metric}"]
    return pd.concat([out.drop(columns=ALL_FEATURE_COLUMNS, errors="ignore"), feature_df], axis=1)
