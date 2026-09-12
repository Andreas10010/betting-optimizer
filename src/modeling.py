from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .feature_engineering import (ALL_FEATURE_COLUMNS, FEATURE_ALIASES, CONTEXT_FEATURE_COLUMNS,
                                      GOAL_MODEL_FEATURE_COLUMNS, SHOT_RATING_FEATURE_COLUMNS, NEW_FEATURE_COLUMNS)
except ImportError:
    from feature_engineering import (ALL_FEATURE_COLUMNS, FEATURE_ALIASES, CONTEXT_FEATURE_COLUMNS,
                                     GOAL_MODEL_FEATURE_COLUMNS, SHOT_RATING_FEATURE_COLUMNS, NEW_FEATURE_COLUMNS)


OUTCOME_MAP = {"H": 0, "D": 1, "A": 2}
ODDS_COLUMNS = ["B365H", "B365D", "B365A"]

FEATURE_GROUPS = {
    "baseline_form": [
        "League",
        "home_recent_form_5",
        "away_recent_form_5",
        "home_recent_form_8",
        "away_recent_form_8",
        "home_recent_goal_diff_5",
        "away_recent_goal_diff_5",
        "home_recent_goal_diff_8",
        "away_recent_goal_diff_8",
        "home_elo",
        "away_elo",
        "elo_diff",
    ],
    "extended_stats": [
        "League",
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
        "home_ppg_l5",
        "away_ppg_l5",
        "home_ppg_l10",
        "away_ppg_l10",
        "home_shot_accuracy_l5",
        "away_shot_accuracy_l5",
        "home_sot_conversion_l5",
        "away_sot_conversion_l5",
        "home_elo",
        "away_elo",
        "elo_diff",
        "home_position",
        "away_position",
        "position_diff",
    ],
    "all_features": ["League", *ALL_FEATURE_COLUMNS],
}


def _canonical_features(columns):
    return list(dict.fromkeys(FEATURE_ALIASES.get(column, column) for column in columns))


FEATURE_GROUPS = {name: _canonical_features(columns) for name, columns in FEATURE_GROUPS.items()}
# Add one family at a time to the same baseline for an interpretable comparison.
FEATURE_GROUPS.update({
    "baseline_context": _canonical_features(FEATURE_GROUPS["baseline_form"] + CONTEXT_FEATURE_COLUMNS),
    "baseline_poisson": _canonical_features(FEATURE_GROUPS["baseline_form"] + GOAL_MODEL_FEATURE_COLUMNS),
    "baseline_shot_ratings": _canonical_features(FEATURE_GROUPS["baseline_form"] + SHOT_RATING_FEATURE_COLUMNS),
    "corrected_legacy": _canonical_features(["League", *[c for c in ALL_FEATURE_COLUMNS if c not in NEW_FEATURE_COLUMNS]]),
})


def build_modeling_data(df: pd.DataFrame, feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    if feature_cols is None:
        feature_cols = FEATURE_GROUPS["all_features"]

    feature_cols = _canonical_features(feature_cols)
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing engineered features: {sorted(missing)}")
    model_df = df.copy()
    model_df["MatchDate"] = pd.to_datetime(model_df["MatchDate"], errors="coerce").dt.normalize()
    # Missing historical stats are imputed within each training fold only.
    model_df = model_df.dropna(subset=["target_label", "MatchDate"]).copy()
    model_df = model_df[model_df["target_label"].isin([0, 1, 2])].sort_values("MatchDate").copy()
    model_df["target_label"] = model_df["target_label"].astype(int)

    return model_df, feature_cols


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
    num_cols = [col for col in X.columns if col not in cat_cols]

    transformers = []

    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                          ("scale", StandardScaler())]),
                num_cols,
            )
        )

    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            )
        )

    return ColumnTransformer(transformers=transformers)


def _time_split(model_df: pd.DataFrame, train_end_year: int = 2022, val_end_year: int = 2024) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_end_year >= val_end_year:
        raise ValueError("train_end_year must precede val_end_year")
    data = model_df.copy()
    data["MatchDate"] = pd.to_datetime(data["MatchDate"], errors="coerce").dt.normalize()
    if data["MatchDate"].isna().any():
        raise ValueError("Time splits require valid match dates")
    data = data.sort_values("MatchDate").reset_index(drop=True)

    train_df = data[data["MatchDate"].dt.year <= train_end_year].copy()
    val_df = data[(data["MatchDate"].dt.year > train_end_year) & (data["MatchDate"].dt.year <= val_end_year)].copy()
    test_df = data[data["MatchDate"].dt.year > val_end_year].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        dates = data["MatchDate"].dropna().dt.normalize().drop_duplicates().sort_values().to_numpy()
        if len(dates) < 3:
            raise ValueError("At least three distinct match dates are needed for disjoint train/validation/test sets")
        train_end = min(max(1, int(len(dates) * .65)), len(dates) - 2)
        val_end = min(max(train_end + 1, int(len(dates) * .85)), len(dates) - 1)
        train_df = data[data["MatchDate"] < dates[train_end]].copy()
        val_df = data[(data["MatchDate"] >= dates[train_end]) & (data["MatchDate"] < dates[val_end])].copy()
        test_df = data[data["MatchDate"] >= dates[val_end]].copy()

    return train_df, val_df, test_df


def _predict_probabilities(pipe, X):
    # A fold may lack one outcome: keep H/D/A columns in their defined order.
    raw = pipe.predict_proba(X)
    probs = np.zeros((len(X), 3))
    probs[:, pipe.named_steps["model"].classes_.astype(int)] = raw
    return probs


def _probability_metrics(y, probs):
    actual = np.eye(3)[np.asarray(y, dtype=int)]
    return {
        "log_loss": float(log_loss(y, probs, labels=[0, 1, 2])),
        "accuracy": float(accuracy_score(y, probs.argmax(axis=1))),
        "brier_score": float(np.mean(np.sum((probs - actual) ** 2, axis=1))),
    }


def _evaluate_model(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    return _probability_metrics(y_test, _predict_probabilities(pipe, X_test))


def calibration_table(y, probs, n_bins=10):
    records = []
    y = np.asarray(y)
    for outcome, label in enumerate(("H", "D", "A")):
        bins = np.minimum((probs[:, outcome] * n_bins).astype(int), n_bins - 1)
        for bucket in range(n_bins):
            selected = bins == bucket
            if selected.any():
                records.append({"outcome": label, "bin": bucket, "count": int(selected.sum()),
                                "mean_probability": float(probs[selected, outcome].mean()),
                                "observed_frequency": float((y[selected] == outcome).mean())})
    return pd.DataFrame(records)


def walk_forward_compare(df, groups=None, n_splits=3):
    """Expanding-date folds on identical rows; final 15% of dates stay untouched.

    Features must be engineered by engineer_features before calling this. Ratings
    may update with completed earlier evaluation matches, as in daily deployment.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be positive")
    groups = FEATURE_GROUPS if groups is None else groups
    union = list(dict.fromkeys(c for columns in groups.values() for c in columns))
    data, _ = build_modeling_data(df, union)
    dates = data["MatchDate"].drop_duplicates().sort_values().to_numpy()
    holdout_start = min(int(len(dates) * .85), len(dates) - 1)
    development_dates = dates[:holdout_start]
    if len(development_dates) < 2 * (n_splits + 1):
        raise ValueError("Too few distinct dates for the requested walk-forward comparison")
    first_eval = len(development_dates) // 2
    blocks = np.array_split(development_dates[first_eval:], n_splits)
    results, calibration = [], []
    for group, columns in groups.items():
        columns = _canonical_features(columns)
        collected_y, collected_probs = [], []
        for fold, block in enumerate(blocks, 1):
            train = data[data["MatchDate"] < block[0]]
            valid = data[data["MatchDate"].isin(block)]
            if train["target_label"].nunique() < 2:
                raise ValueError("Each training fold needs at least two result classes")
            pipe = Pipeline([("preprocessor", _build_preprocessor(train[columns])),
                             ("model", LogisticRegression(max_iter=3000))])
            pipe.fit(train[columns], train["target_label"])
            probs = _predict_probabilities(pipe, valid[columns])
            results.append({"feature_group": group, "fold": fold, "features": len(columns),
                            "train_rows": len(train), "eval_rows": len(valid),
                            "train_end": train["MatchDate"].max(), "eval_start": block[0], "eval_end": block[-1],
                            "holdout_start": dates[holdout_start],
                            **_probability_metrics(valid["target_label"], probs)})
            collected_y.extend(valid["target_label"])
            collected_probs.append(probs)
        table = calibration_table(collected_y, np.concatenate(collected_probs))
        table["feature_group"] = group
        calibration.append(table)
    return pd.DataFrame(results), pd.concat(calibration, ignore_index=True)


def train_baseline(
    model_df: pd.DataFrame,
    feature_cols: list[str],
    train_end_year: int = 2022,
    val_end_year: int = 2024,
) -> tuple[Pipeline, dict[str, float], dict[str, int | pd.DataFrame]]:
    train_df, val_df, test_df = _time_split(model_df, train_end_year, val_end_year)

    X_train = train_df[feature_cols]
    y_train = train_df["target_label"]
    X_test = test_df[feature_cols]
    y_test = test_df["target_label"]

    preprocessor = _build_preprocessor(X_train)
    estimator = LogisticRegression(max_iter=3000)
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    if y_train.nunique() < 2:
        raise ValueError("Training data needs at least two result classes")
    pipe.fit(X_train, y_train)

    metrics = _evaluate_model(pipe, X_test, y_test)

    split_info = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }

    return pipe, metrics, split_info


def train_ml_model(
    model_df: pd.DataFrame,
    feature_cols: list[str],
    train_end_year: int = 2022,
    val_end_year: int = 2024,
) -> tuple[Pipeline, dict[str, float], dict[str, int | pd.DataFrame]]:
    train_df, val_df, test_df = _time_split(model_df, train_end_year, val_end_year)

    X_train = train_df[feature_cols]
    y_train = train_df["target_label"]
    X_test = test_df[feature_cols]
    y_test = test_df["target_label"]

    preprocessor = _build_preprocessor(X_train)
    estimator = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced_subsample",
    )
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    if y_train.nunique() < 2:
        raise ValueError("Training data needs at least two result classes")
    pipe.fit(X_train, y_train)

    metrics = _evaluate_model(pipe, X_test, y_test)

    split_info = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }

    return pipe, metrics, split_info


def compute_feature_importances(model_pipe: Pipeline) -> pd.DataFrame:
    model = model_pipe.named_steps["model"]
    preprocessor = model_pipe.named_steps["preprocessor"]

    transformed_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
    elif hasattr(model, "importance_"):
        importance_values = model.importance_
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    feature_df = pd.DataFrame(
        {
            "feature": transformed_names,
            "importance": importance_values,
        }
    )
    feature_df = feature_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return feature_df


def has_bookmaker_odds(df: pd.DataFrame) -> bool:
    return all(col in df.columns for col in ODDS_COLUMNS)


def compute_bookmaker_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["market_prob_home", "market_prob_draw", "market_prob_away"]
    if not has_bookmaker_odds(df):
        return pd.DataFrame(np.nan, index=df.index, columns=columns)
    odds = df[ODDS_COLUMNS].apply(pd.to_numeric, errors="coerce")
    valid = (np.isfinite(odds) & odds.gt(1)).all(axis=1)
    implied = (1 / odds).where(valid, np.nan)
    implied = implied.div(implied.sum(axis=1), axis=0)
    implied.columns = columns
    return implied


def summarize_edge(model_pipe: Pipeline, feature_cols: list[str], eval_df: pd.DataFrame) -> pd.DataFrame | None:
    if not has_bookmaker_odds(eval_df):
        return None
    market_probs = compute_bookmaker_probabilities(eval_df)

    X_eval = eval_df[feature_cols]
    model_probs = _predict_probabilities(model_pipe, X_eval)
    prob_df = pd.DataFrame(model_probs, columns=["prob_home", "prob_draw", "prob_away"])

    result = pd.concat([eval_df.reset_index(drop=True), prob_df.reset_index(drop=True), market_probs.reset_index(drop=True)], axis=1)
    result["edge_home"] = result["prob_home"] - result["market_prob_home"]
    result["edge_draw"] = result["prob_draw"] - result["market_prob_draw"]
    result["edge_away"] = result["prob_away"] - result["market_prob_away"]
    result["best_edge"] = result[["edge_home", "edge_draw", "edge_away"]].max(axis=1)

    return result
