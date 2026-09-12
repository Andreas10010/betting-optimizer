import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.feature_engineering import FEATURE_ALIASES, ALL_FEATURE_COLUMNS
from src.modeling import (FEATURE_GROUPS, _time_split, _build_preprocessor, _predict_probabilities,
                          build_modeling_data, walk_forward_compare, compute_bookmaker_probabilities,
                          summarize_edge, calibration_table)


def test_feature_registry_is_complete_and_has_no_alias_duplicates():
    for columns in FEATURE_GROUPS.values():
        assert len(columns) == len(set(columns))
        assert not set(columns) & set(FEATURE_ALIASES)
        assert set(columns) <= {'League', *ALL_FEATURE_COLUMNS}
    canonical = {FEATURE_ALIASES.get(c, c) for c in ALL_FEATURE_COLUMNS}
    assert set(FEATURE_GROUPS['all_features']) == {'League', *canonical}


def test_missing_features_preserve_rows_and_imputation_is_training_only():
    data = pd.DataFrame({'MatchDate': pd.date_range('2023-01-01', periods=4),
                         'target_label': [0, 1, 2, np.nan], 'x': [1, 3, np.nan, 10000],
                         'League': ['L'] * 4})
    model, columns = build_modeling_data(data, ['League', 'x'])
    assert len(model) == 3
    pre = _build_preprocessor(model.iloc[:2][columns])
    pre.fit(model.iloc[:2][columns])
    assert pre.named_transformers_['num'].named_steps['imputer'].statistics_[0] == 2
    assert np.isfinite(pre.transform(model[columns])).all()


def test_time_splits_keep_dates_together_and_are_disjoint():
    data = pd.DataFrame({'MatchDate': list(pd.date_range('2023-01-01', periods=10)) * 4})
    train, val, test = _time_split(data)
    assert train.MatchDate.max() < val.MatchDate.min()
    assert val.MatchDate.max() < test.MatchDate.min()
    assert len(train) + len(val) + len(test) == len(data)
    with pytest.raises(ValueError, match='three distinct'):
        _time_split(data.iloc[:2])
    with pytest.raises(ValueError, match='precede'):
        _time_split(data, train_end_year=2024, val_end_year=2022)


def test_two_class_training_keeps_three_probability_columns():
    X = pd.DataFrame({'x': [0, 1, 2, 3]})
    pipe = Pipeline([('preprocessor', _build_preprocessor(X)), ('model', LogisticRegression())])
    pipe.fit(X, [0, 2, 0, 2])
    probs = _predict_probabilities(pipe, X)
    assert probs.shape == (4, 3)
    assert (probs[:, 1] == 0).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1)


def test_walk_forward_uses_same_rows_and_reserves_final_holdout():
    data = pd.DataFrame({'MatchDate': pd.date_range('2023-01-01', periods=90),
                         'target_label': np.arange(90) % 3, 'x': np.arange(90) % 4,
                         'z': [np.nan] * 80 + list(range(10))})
    scores, calibration = walk_forward_compare(data, {'small': ['x'], 'large': ['x', 'z']})
    assert len(scores) == 6
    assert (scores.train_end < scores.eval_start).all()
    assert (scores.eval_end < scores.holdout_start).all()
    assert scores.groupby('fold').eval_rows.nunique().eq(1).all()
    assert calibration.groupby(['feature_group', 'outcome'])['count'].sum().nunique() == 1
    assert np.isfinite(scores[['log_loss', 'brier_score', 'accuracy']]).all().all()


def test_odds_alignment_and_no_fabricated_bookmaker():
    data = pd.DataFrame({'B365H': [2, np.nan, 4], 'B365D': [3, 3, 4], 'B365A': [4, 4, 2]}, index=[5, 7, 9])
    probs = compute_bookmaker_probabilities(data)
    assert probs.index.tolist() == [5, 7, 9]
    assert probs.loc[7].isna().all()
    assert probs.loc[9, 'market_prob_home'] == .25
    assert summarize_edge(None, [], pd.DataFrame()) is None


def test_calibration_bin_boundaries():
    table = calibration_table([0, 1, 2], np.eye(3))
    assert table.groupby('outcome')['count'].sum().eq(3).all()
    assert (table.mean_probability == table.observed_frequency).all()
