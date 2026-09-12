"""Chronological logistic experiments, with feature screening confined to train."""
from __future__ import annotations

from itertools import combinations
from math import comb
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline

try:
    from .feature_diagnostics import OUTCOMES, discrimination_metrics
    from .modeling import (FEATURE_GROUPS, _build_preprocessor, _canonical_features,
                           _predict_probabilities, _probability_metrics, _time_split,
                           build_modeling_data, calibration_table)
except ImportError:
    from feature_diagnostics import OUTCOMES, discrimination_metrics
    from modeling import (FEATURE_GROUPS, _build_preprocessor, _canonical_features,
                          _predict_probabilities, _probability_metrics, _time_split,
                          build_modeling_data, calibration_table)


def experiment_splits(df, columns=None):
    columns = _canonical_features(FEATURE_GROUPS['all_features'] if columns is None else columns)
    data, _ = build_modeling_data(df, columns)
    train, validation, test = _time_split(data)
    if train.target_label.nunique() < 2:
        raise ValueError('Train behöver innehålla minst två utfall.')
    return {'Train': train, 'Validering': validation, 'Test': test}


def _fit(train, columns):
    pipe = Pipeline([('preprocessor', _build_preprocessor(train[columns])),
                     ('model', LogisticRegression(C=1.0, max_iter=3000))])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        pipe.fit(train[columns], train.target_label)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return pipe, converged


def _metrics(y, probs):
    return {**_probability_metrics(y, probs), **discrimination_metrics(y, probs)}


def screen_training_features(train, columns):
    """Fit on the first 80% of train dates, score on its last 20%; no outer test."""
    dates = train.MatchDate.drop_duplicates().sort_values().to_numpy()
    if len(dates) < 5:
        raise ValueError('Featurefiltret behöver minst fem olika datum inom train.')
    cutoff = dates[min(len(dates) - 1, max(1, int(len(dates) * .8)))]
    fit = train[train.MatchDate < cutoff]
    valid = train[train.MatchDate >= cutoff]
    if fit.target_label.nunique() < 2:
        raise ValueError('Den tidiga delen av train behöver minst två utfall för featurefiltret.')
    priors = np.bincount(fit.target_label, minlength=3) / len(fit)
    dummy = np.tile(priors, (len(valid), 1))
    dummy_metrics = _metrics(valid.target_label, dummy)
    records = []
    for name in _canonical_features(columns):
        status = 'OK'
        if fit[name].nunique(dropna=True) <= 1:
            probs = dummy
            status = 'Konstant eller saknad'
        else:
            pipe, converged = _fit(fit, [name])
            probs = _predict_probabilities(pipe, valid[[name]])
            if not converged:
                status = 'Ej konvergerad'
        metrics = _metrics(valid.target_label, probs)
        records.append({'feature': name, 'screen_gini': metrics['gini_macro'],
                        'screen_log_loss_gain': dummy_metrics['log_loss'] - metrics['log_loss'],
                        'screen_status': status, 'fit_end': fit.MatchDate.max(),
                        'screen_start': valid.MatchDate.min(), 'screen_end': valid.MatchDate.max(),
                        'fit_rows': len(fit), 'screen_rows': len(valid)})
    return pd.DataFrame(records)


def screening_for_data(df):
    columns = FEATURE_GROUPS['all_features']
    return screen_training_features(experiment_splits(df)['Train'], columns)


def eligible_features(screening, threshold=.02):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Gini-gränsen måste vara mellan 0 och 1.')
    good = screening.screen_gini.gt(threshold) & screening.screen_status.eq('OK')
    return screening.loc[good, 'feature'].tolist()


def generate_combinations(features, n_simulations=50, min_features=5, max_features=20, seed=42):
    names = sorted(_canonical_features(features))
    if n_simulations < 1 or not 1 <= min_features <= max_features <= len(names):
        raise ValueError(f'Välj minst en simulering och 1 ≤ min features ≤ max features ≤ {len(names)}.')
    total = sum(comb(len(names), k) for k in range(min_features, max_features + 1))
    count = min(n_simulations, total)
    rng = np.random.default_rng(seed)
    if total <= 10000:
        possibilities = [tuple(c) for k in range(min_features, max_features + 1) for c in combinations(names, k)]
        return [possibilities[i] for i in rng.choice(total, size=count, replace=False)]
    picked, seen = [], set()
    while len(picked) < count:
        size = int(rng.integers(min_features, max_features + 1))
        candidate = tuple(sorted(rng.choice(names, size=size, replace=False).tolist()))
        if candidate not in seen:
            seen.add(candidate)
            picked.append(candidate)
    return picked


def run_combination_search(df, features, n_simulations=50, min_features=5, max_features=20,
                           seed=42, progress=None):
    """Fixed-seed random subsets, identical dates; no feedback from test to sampling."""
    subsets = generate_combinations(features, n_simulations, min_features, max_features, seed)
    parts = experiment_splits(df, features)
    train = parts['Train']
    priors = np.bincount(train.target_label, minlength=3) / len(train)
    records = []
    for number, subset in enumerate(subsets, 1):
        columns = list(subset)
        pipe, converged = _fit(train, columns)
        record = {'simulation': number, 'n_features': len(columns), 'features': columns,
                  'status': 'OK' if converged else 'Ej konvergerad'}
        for partition, data in parts.items():
            prefix = {'Train': 'train', 'Validering': 'validation', 'Test': 'test'}[partition]
            probs = _predict_probabilities(pipe, data[columns])
            metrics = _metrics(data.target_label, probs)
            dummy = _metrics(data.target_label, np.tile(priors, (len(data), 1)))
            for name in ('gini_macro', 'gini_home', 'gini_draw', 'gini_away', 'log_loss', 'brier_score', 'accuracy'):
                record[f'{prefix}_{name}'] = metrics[name]
            record[f'{prefix}_log_loss_gain'] = dummy['log_loss'] - metrics['log_loss']
            record[f'{prefix}_rows'] = len(data)
        records.append(record)
        if progress:
            progress(number, len(subsets))
    return pd.DataFrame(records)


def rank_combinations(results, partition='test', metric='gini_macro'):
    if partition not in ('train', 'validation', 'test') or metric not in ('gini_macro', 'log_loss', 'brier_score'):
        raise ValueError('Okänd rangordning.')
    # Fits that did not converge remain visible in raw results but cannot win.
    valid = results[results.status.eq('OK') & results[f'{partition}_{metric}'].notna()]
    return valid.sort_values([f'{partition}_{metric}', 'simulation'],
                             ascending=[metric != 'gini_macro', True]).reset_index(drop=True)


def logistic_report(df, features):
    """Same fitted classifier on train/validation/test; retain per-match probabilities."""
    columns = _canonical_features(features)
    if not columns:
        raise ValueError('Modellen behöver minst en feature.')
    parts = experiment_splits(df, columns)
    train = parts['Train']
    pipe, converged = _fit(train, columns)
    priors = np.bincount(train.target_label, minlength=3) / len(train)
    metrics_rows, outcomes, calibration, curves = [], [], [], []
    predictions = {}
    for partition, data in parts.items():
        probs = _predict_probabilities(pipe, data[columns])
        dummy_probs = np.tile(priors, (len(data), 1))
        for name, probabilities in (('Dummy', dummy_probs), ('LG', probs)):
            metrics_rows.append({'partition': partition, 'model': name, 'rows': len(data),
                                 **_metrics(data.target_label, probabilities)})
        for outcome, (label, title) in OUTCOMES.items():
            outcomes.append({'partition': partition, 'outcome': title, 'dummy_probability': priors[label],
                             'model_probability': probs[:, label].mean(),
                             'observed_frequency': data.target_label.eq(label).mean(),
                             'predicted_share': (probs.argmax(axis=1) == label).mean()})
            binary = data.target_label.eq(label).astype(int)
            if binary.nunique() == 2:
                fpr, tpr, _ = roc_curve(binary, probs[:, label])
                # Only downsample the plotted line; Gini uses every test observation.
                points = np.unique(np.linspace(0, len(fpr) - 1, min(300, len(fpr)), dtype=int))
                curves.extend({'partition': partition, 'outcome': title, 'fpr': fpr[i], 'tpr': tpr[i]} for i in points)
        table = calibration_table(data.target_label, probs)
        table['partition'] = partition
        calibration.append(table)
        match_cols = [c for c in ('MatchDate', 'League', 'HomeTeam', 'AwayTeam', 'target_label') if c in data]
        prediction = data[match_cols].reset_index(drop=True).copy()
        for outcome, (label, _) in OUTCOMES.items():
            prediction[f'prob_{outcome}'] = probs[:, label]
            prediction[f'dummy_{outcome}'] = priors[label]
        prediction['confidence'] = probs.max(axis=1)
        prediction['predicted_label'] = probs.argmax(axis=1)
        predictions[partition] = prediction
    return {'features': columns, 'converged': converged, 'metrics': pd.DataFrame(metrics_rows),
            'outcomes': pd.DataFrame(outcomes), 'calibration': pd.concat(calibration, ignore_index=True),
            'roc': pd.DataFrame(curves, columns=['partition', 'outcome', 'fpr', 'tpr']),
            'predictions': predictions,
            'splits': pd.DataFrame([{'partition': name, 'start': part.MatchDate.min(),
                                    'end': part.MatchDate.max(), 'rows': len(part)} for name, part in parts.items()])}


def confidence_summary(report, threshold=.60, enabled=True):
    """Summarize fixed pre-match predictions after a label-independent filter.

    Dummy priors remain those fitted on all training rows, even when no rows
    survive the filter. Empty selections have undefined model/outcome averages.
    """
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Säkerhetsgränsen måste vara mellan 0 och 1.')
    selected, coverage = {}, []
    prob_columns = [f'prob_{name}' for name in OUTCOMES]
    for partition in ('Train', 'Test'):
        data = report['predictions'][partition]
        confidence = data[prob_columns].max(axis=1)
        kept = data.loc[confidence.ge(threshold)].copy() if enabled else data.copy()
        selected[partition] = kept
        coverage.append({'Period': partition, 'Totalt': len(data), 'Kvar': len(kept),
                         'Bortfiltrerade': len(data) - len(kept),
                         'Andel kvar': len(kept) / len(data) if len(data) else np.nan,
                         'Andel rätt': kept.predicted_label.eq(kept.target_label).mean() if len(kept) else np.nan})
    training = report['predictions']['Train']
    rows = []
    for name, (label, title) in OUTCOMES.items():
        row = {'Utfall': title, 'Dummy (hela train)': training[f'dummy_{name}'].iloc[0]}
        for partition in ('Train', 'Test'):
            kept = selected[partition]
            row[f'LG medel, {partition.lower()}'] = kept[f'prob_{name}'].mean()
            row[f'Observerat, {partition.lower()}'] = kept.target_label.eq(label).mean() if len(kept) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index('Utfall'), pd.DataFrame(coverage), selected
