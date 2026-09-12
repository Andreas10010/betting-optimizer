"""One logistic model per feature, evaluated against a training-prior dummy."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

try:
    from .modeling import (FEATURE_GROUPS, _build_preprocessor, _canonical_features,
                           _predict_probabilities, _probability_metrics, _time_split,
                           build_modeling_data)
except ImportError:
    from modeling import (FEATURE_GROUPS, _build_preprocessor, _canonical_features,
                          _predict_probabilities, _probability_metrics, _time_split,
                          build_modeling_data)


OUTCOMES = {"home": (0, "Hemmavinst (H)"), "draw": (1, "Oavgjort (D)"), "away": (2, "Bortavinst (A)")}


def discrimination_metrics(y, probabilities):
    """One-vs-rest Gini = 2*AUC-1; undefined for a single-class test target."""
    y = np.asarray(y, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    metrics = {}
    for name, (label, _) in OUTCOMES.items():
        binary = (y == label).astype(int)
        p = probabilities[:, label]
        auc = float(roc_auc_score(binary, p)) if len(np.unique(binary)) == 2 else np.nan
        metrics[f"gini_{name}"] = 2 * auc - 1
        metrics[f"log_loss_{name}"] = float(log_loss(binary, np.column_stack([1 - p, p]), labels=[0, 1]))
        metrics[f"brier_{name}"] = float(np.mean((p - binary) ** 2))
    # Do not silently average fewer than three outcomes.
    metrics["gini_macro"] = float(np.mean([metrics[f"gini_{name}"] for name in OUTCOMES]))
    return metrics


def evaluate_individual_features(df, feature_cols=None, train_end_year=2022, val_end_year=2024):
    """Use identical chronological test rows for every univariate classifier.

    Only training labels determine the dummy probabilities. Imputation, scaling,
    encoding and logistic coefficients are also learned on training rows only.
    This measures standalone predictive value, not incremental multivariate value.
    """
    columns = _canonical_features(FEATURE_GROUPS["all_features"] if feature_cols is None else feature_cols)
    if not columns:
        raise ValueError("Välj minst en feature")
    data, _ = build_modeling_data(df, columns)
    train, validation, test = _time_split(data, train_end_year, val_end_year)
    y_train, y_test = train["target_label"], test["target_label"]
    if y_train.nunique() < 2:
        raise ValueError("Träningsdata behöver innehålla minst två olika matchutfall")

    priors = np.bincount(y_train, minlength=3) / len(train)
    dummy_probs = np.tile(priors, (len(test), 1))
    dummy = {**_probability_metrics(y_test, dummy_probs), **discrimination_metrics(y_test, dummy_probs)}
    outcome_rows = []
    for name, (label, title) in OUTCOMES.items():
        outcome_rows.append({"outcome": title, "key": name, "train_count": int((y_train == label).sum()),
                             "dummy_probability": float(priors[label]),
                             "test_frequency": float((y_test == label).mean()),
                             "test_count": int((y_test == label).sum()),
                             "dummy_gini": dummy[f"gini_{name}"],
                             "dummy_log_loss": dummy[f"log_loss_{name}"],
                             "dummy_brier": dummy[f"brier_{name}"]})

    scores = []
    for feature in columns:
        X_train, X_test = train[[feature]], test[[feature]]
        status = "OK"
        if X_train[feature].nunique(dropna=True) <= 1:
            probabilities = dummy_probs.copy()
            status = "Saknas i train" if X_train[feature].isna().all() else "Konstant i train"
        else:
            pipe = Pipeline([("preprocessor", _build_preprocessor(X_train)),
                             ("model", LogisticRegression(max_iter=3000, C=1.0))])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                pipe.fit(X_train, y_train)
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                status = "Optimeringen konvergerade inte"
            probabilities = _predict_probabilities(pipe, X_test)
        if y_train.nunique() < 3:
            status += "; ett utfall saknas i train"
        metrics = {**_probability_metrics(y_test, probabilities), **discrimination_metrics(y_test, probabilities)}
        record = {"feature": feature, "status": status, **metrics,
                  "train_missing": float(X_train[feature].isna().mean()),
                  "test_missing": float(X_test[feature].isna().mean()),
                  "train_rows": len(train), "test_rows": len(test),
                  "log_loss_gain": dummy["log_loss"] - metrics["log_loss"],
                  "brier_gain": dummy["brier_score"] - metrics["brier_score"],
                  "accuracy_gain": metrics["accuracy"] - dummy["accuracy"]}
        for name in OUTCOMES:
            record[f"log_loss_gain_{name}"] = dummy[f"log_loss_{name}"] - metrics[f"log_loss_{name}"]
            record[f"brier_gain_{name}"] = dummy[f"brier_{name}"] - metrics[f"brier_{name}"]
        scores.append(record)

    split = {"train_start": train.MatchDate.min(), "train_end": train.MatchDate.max(),
             "validation_start": validation.MatchDate.min(), "validation_end": validation.MatchDate.max(),
             "test_start": test.MatchDate.min(), "test_end": test.MatchDate.max(),
             "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test)}
    return (pd.DataFrame(scores).sort_values(["log_loss_gain", "feature"], ascending=[False, True]).reset_index(drop=True),
            pd.DataFrame(outcome_rows), dummy, split)
