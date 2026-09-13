"""Logistisk modell tränad på all Premier League-historik före 2023/24."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LEAGUE = "Premier League"
TEST_SEASON = "2023/24"
ARTIFACT_NAME = "premier_league_lg_all_history.joblib"
CLASS_ORDER = ["Förlust", "Oavgjort", "Vinst"]
CATEGORICAL_COLUMNS = ["Venue"]
PROBABILITY_COLUMNS = ["ProbabilityLoss", "ProbabilityDraw", "ProbabilityWin"]

FEATURE_SETS = {
    "form": [
        "ppg_total_l5", "ppg_home_l5", "ppg_away_l5",
        "goal_diff_total_l5", "goal_diff_home_l5", "goal_diff_away_l5",
        "shots_for_total_l5", "shots_against_total_l5",
        "sot_for_total_l5", "sot_against_total_l5",
    ],
    "form_strength": [
        "ppg_total_l5", "ppg_home_l5", "ppg_away_l5",
        "goal_diff_total_l5", "goal_diff_home_l5", "goal_diff_away_l5",
        "shots_for_total_l5", "shots_against_total_l5",
        "sot_for_total_l5", "sot_against_total_l5",
        "position_advantage", "elo_difference",
    ],
    "form_strength_model": [
        "ppg_total_l5", "ppg_home_l5", "ppg_away_l5",
        "goal_diff_total_l5", "goal_diff_home_l5", "goal_diff_away_l5",
        "shots_for_total_l5", "shots_against_total_l5",
        "sot_for_total_l5", "sot_against_total_l5",
        "position_advantage", "elo_difference",
        "team_elo_expected", "expected_goal_difference",
        "poisson_prob_win", "poisson_prob_loss",
    ],
}

BEST_LOGISTIC = {
    "feature_set": "form_strength_model",
    "C": 0.03,
    "class_weight": None,
}
# Samma hyperparametrar som sökgriden i notebooks/premier_league_all_history.ipynb.
# Slutmodellen tränas på alla säsonger före TEST_SEASON.

FEATURE_LABELS = {
    "ppg_total_l5": "Poäng per match, 5 matcher",
    "ppg_home_l5": "Poäng per match hemma, 5 matcher",
    "ppg_away_l5": "Poäng per match borta, 5 matcher",
    "goal_diff_total_l5": "Målskillnad, 5 matcher",
    "goal_diff_home_l5": "Målskillnad hemma, 5 matcher",
    "goal_diff_away_l5": "Målskillnad borta, 5 matcher",
    "shots_for_total_l5": "Skott för, 5 matcher",
    "shots_against_total_l5": "Skott mot, 5 matcher",
    "sot_for_total_l5": "Skott på mål för, 5 matcher",
    "sot_against_total_l5": "Skott på mål mot, 5 matcher",
    "position_advantage": "Tabellfördel",
    "elo_difference": "Elo-skillnad",
    "team_elo_expected": "Förväntad Elo-poäng",
    "expected_goal_difference": "Förväntad målskillnad",
    "poisson_prob_win": "Poisson-sannolikhet vinst",
    "poisson_prob_loss": "Poisson-sannolikhet förlust",
    "Venue_Hemma": "Hemma",
    "intercept": "Basnivå",
}


class CorrelationPruner(BaseEstimator, TransformerMixin):
    """Behåll en representant ur grupper med hög absolut träningskorrelation."""

    def __init__(self, threshold=0.90):
        self.threshold = threshold

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        correlation = np.nan_to_num(np.corrcoef(values, rowvar=False), nan=0.0)
        kept = []
        for index in range(values.shape[1]):
            if all(abs(correlation[index, previous]) < self.threshold for previous in kept):
                kept.append(index)
        self.kept_indices_ = np.asarray(kept, dtype=int)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.kept_indices_]

    def get_feature_names_out(self, input_features=None):
        names = np.asarray(input_features, dtype=object)
        return names[self.kept_indices_]


def numeric_feature_columns(feature_set=None):
    name = BEST_LOGISTIC["feature_set"] if feature_set is None else feature_set
    return list(FEATURE_SETS[name])


def model_input_columns(feature_set=None):
    return CATEGORICAL_COLUMNS + numeric_feature_columns(feature_set)


def logistic_pipeline(numeric_columns=None, c_value=None, class_weight=None):
    numeric_columns = numeric_feature_columns() if numeric_columns is None else list(numeric_columns)
    c_value = BEST_LOGISTIC["C"] if c_value is None else c_value
    class_weight = BEST_LOGISTIC["class_weight"] if class_weight is None else class_weight
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("correlation_pruner", CorrelationPruner(threshold=0.90)),
            ("scaler", StandardScaler()),
        ]), numeric_columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL_COLUMNS),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", LogisticRegression(max_iter=5_000, C=c_value, class_weight=class_weight)),
    ])


def split_train_test(team_data, test_season=TEST_SEASON):
    """Train = alla säsonger före test_season, test = den orörda testsäsongen."""
    data = team_data.dropna(subset=["Target"]).copy()
    data = data[data["Target"].isin(CLASS_ORDER)].copy()
    seasons = data["Season"].astype(str)
    train = data.loc[seasons.lt(str(test_season))].sort_values(["MatchDate", "MatchId", "Team"]).reset_index(drop=True)
    test = data.loc[seasons.eq(str(test_season))].sort_values(["MatchDate", "MatchId", "Team"]).reset_index(drop=True)
    if train.empty:
        raise ValueError(f"Ingen träningsdata före säsong {test_season}.")
    if test.empty:
        raise ValueError(f"Ingen testdata för säsong {test_season}.")
    if train["Target"].nunique() < 2:
        raise ValueError("Train behöver innehålla minst två utfall.")
    return train, test


def artifact_path(root=None) -> Path:
    base = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    return base / "data" / ARTIFACT_NAME


def save_trained_model(path, bundle):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_trained_model(path):
    bundle = joblib.load(path)
    if not {"model", "team_data", "config"} <= set(bundle):
        raise ValueError("Modellfilen saknar model, team_data eller config.")
    saved = bundle["config"]
    if saved.get("feature_set") != BEST_LOGISTIC["feature_set"] or float(saved.get("C")) != float(BEST_LOGISTIC["C"]):
        raise ValueError("Sparad modell har en annan konfiguration än den valda LG-modellen.")
    return bundle


def class_shares(labels):
    shares = pd.Series(labels).value_counts(normalize=True)
    return np.array([float(shares.get(label, 0.0)) for label in CLASS_ORDER])


def train_all_history_model(team_data, test_season=TEST_SEASON):
    train, test = split_train_test(team_data, test_season=test_season)
    model = fit_best_logistic(train)
    predicted_test = add_predictions(model, test)
    return {
        "model": model,
        "team_data": team_data,
        "test": predicted_test,
        "metrics": evaluate_predictions(predicted_test, dummy_priors=class_shares(train["Target"])),
        "config": dict(BEST_LOGISTIC),
        "league": LEAGUE,
        "test_season": test_season,
        "train_start": str(train["Season"].min()),
        "train_end": str(train["Season"].max()),
        "train_rows": len(train),
    }


def probabilities_in_order(model, frame, columns=None):
    columns = model_input_columns() if columns is None else columns
    raw = model.predict_proba(frame[columns])
    fitted = list(model.named_steps["model"].classes_)
    return np.column_stack([raw[:, fitted.index(label)] for label in CLASS_ORDER])


def add_predictions(model, frame, columns=None):
    columns = model_input_columns() if columns is None else columns
    result = frame.copy()
    probabilities = probabilities_in_order(model, result, columns)
    predicted = np.array(CLASS_ORDER)[probabilities.argmax(axis=1)]
    result["Predicted"] = predicted
    result["ProbabilityPredicted"] = probabilities.max(axis=1)
    result["ProbabilityLoss"] = probabilities[:, 0]
    result["ProbabilityDraw"] = probabilities[:, 1]
    result["ProbabilityWin"] = probabilities[:, 2]
    result["Correct"] = result["Target"].eq(result["Predicted"])
    return result


def fit_best_logistic(train_df, feature_set=None, c_value=None, class_weight=None):
    columns = model_input_columns(feature_set)
    missing = [name for name in columns if name not in train_df.columns]
    if missing:
        raise ValueError(f"Saknade modellfeatures: {missing}")
    pipe = logistic_pipeline(numeric_feature_columns(feature_set), c_value, class_weight)
    pipe.fit(train_df[columns], train_df["Target"])
    return pipe


def evaluate_predictions(frame, dummy_priors=None):
    probabilities = frame[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    accuracy = float(accuracy_score(frame["Target"], frame["Predicted"]))
    loss = float(log_loss(frame["Target"], probabilities, labels=CLASS_ORDER))
    gini = {}
    for index, label in enumerate(CLASS_ORDER):
        binary = frame["Target"].eq(label).astype(int)
        auc = float(roc_auc_score(binary, probabilities[:, index])) if binary.nunique() == 2 else np.nan
        gini[label] = 2 * auc - 1
    priors = class_shares(frame["Target"]) if dummy_priors is None else np.asarray(dummy_priors, dtype=float)
    dummy_probs = np.tile(priors, (len(frame), 1))
    dummy_pred = np.full(len(frame), CLASS_ORDER[int(priors.argmax())])
    return {
        "rows": len(frame),
        "accuracy": accuracy,
        "log_loss": loss,
        "gini_macro": float(np.mean(list(gini.values()))),
        "gini": gini,
        "dummy_accuracy": float(accuracy_score(frame["Target"], dummy_pred)),
        "dummy_log_loss": float(log_loss(frame["Target"], dummy_probs, labels=CLASS_ORDER)),
        "dummy_priors": priors,
    }


def cleaned_feature_name(name):
    return str(name).replace("numeric__", "").replace("categorical__", "")


def feature_label(name):
    cleaned = cleaned_feature_name(name)
    return FEATURE_LABELS.get(cleaned, cleaned)


def kept_numeric_features(model, numeric_columns=None):
    numeric_columns = numeric_feature_columns() if numeric_columns is None else list(numeric_columns)
    pruner = model.named_steps["preprocessing"].named_transformers_["numeric"].named_steps["correlation_pruner"]
    return [numeric_columns[index] for index in pruner.kept_indices_]


def _as_frame(row):
    if isinstance(row, pd.DataFrame):
        return row.copy()
    return pd.DataFrame([row])


def explain_row(model, row, outcome=None, columns=None):
    """Bidrag till log-odds för ett utfall, efter imputering, korrelationsrensning och skalning."""
    columns = model_input_columns() if columns is None else columns
    frame = _as_frame(row)
    preprocessor = model.named_steps["preprocessing"]
    classifier = model.named_steps["model"]
    transformed = np.asarray(preprocessor.transform(frame[columns]), dtype=float).ravel()
    names = [cleaned_feature_name(name) for name in preprocessor.get_feature_names_out()]
    classes = list(classifier.classes_)
    probabilities = probabilities_in_order(model, frame, columns)[0]
    predicted = CLASS_ORDER[int(np.argmax(probabilities))]
    outcome = predicted if outcome is None else outcome
    class_index = classes.index(outcome)
    coefficients = classifier.coef_[class_index]
    intercept = float(classifier.intercept_[class_index])
    contributions = coefficients * transformed
    table = pd.DataFrame({
        "feature": names,
        "label": [feature_label(name) for name in names],
        "scaled_value": transformed,
        "coefficient": coefficients,
        "contribution": contributions,
    })
    table["abs_contribution"] = table["contribution"].abs()
    table = table.sort_values("abs_contribution", ascending=False).reset_index(drop=True)
    logit = intercept + float(contributions.sum())
    return {
        "outcome": outcome,
        "predicted": predicted,
        "probabilities": dict(zip(CLASS_ORDER, map(float, probabilities))),
        "intercept": intercept,
        "logit": logit,
        "contributions": table,
        "raw_values": {name: frame.iloc[0][name] for name in columns},
    }
