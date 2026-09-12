import numpy as np
import pandas as pd
import pytest

from src.feature_catalog import build_feature_catalog
from src.feature_engineering import ALL_FEATURE_COLUMNS, FEATURE_ALIASES
from src.feature_diagnostics import evaluate_individual_features, discrimination_metrics


def sample():
    y_train = np.repeat([0, 1, 2], [60, 20, 10])
    y_test = np.repeat([0, 1, 2], [10, 20, 60])
    y = np.concatenate([y_train, [0, 1, 2], y_test])
    dates = [*pd.date_range('2020-01-01', periods=90), *pd.date_range('2023-01-01', periods=3),
             *pd.date_range('2025-01-01', periods=90)]
    return pd.DataFrame({'MatchDate': dates, 'target_label': y, 'signal': y.astype(float),
                         'constant': 7., 'missing': np.nan, 'category': np.where(y == 0, 'a', 'b')})


def test_dummy_is_training_prior_and_every_feature_has_the_same_test_rows():
    scores, outcomes, dummy, split = evaluate_individual_features(sample(), ['signal', 'constant', 'missing', 'category'])
    np.testing.assert_allclose(outcomes.dummy_probability, np.array([60, 20, 10]) / 90)
    np.testing.assert_allclose(outcomes.test_frequency, np.array([10, 20, 60]) / 90)
    assert outcomes.dummy_gini.eq(0).all()
    assert dummy['accuracy'] == pytest.approx(10 / 90)
    assert scores.test_rows.eq(90).all()
    assert scores.train_rows.eq(90).all()
    assert split['train_end'] < split['validation_start'] < split['test_start']
    indexed = scores.set_index('feature')
    assert indexed.loc['signal', 'log_loss_gain'] > 0
    assert indexed.loc['signal', 'gini_macro'] > .9
    for feature in ['constant', 'missing']:
        assert indexed.loc[feature, 'log_loss_gain'] == 0
        assert indexed.loc[feature, 'gini_macro'] == 0
        assert indexed.loc[feature, 'brier_gain'] == 0
    assert indexed.loc['missing', 'test_missing'] == 1


def test_perfect_constant_and_reversed_gini_and_absent_test_outcomes():
    y = np.array([0, 1, 2, 0, 1, 2])
    perfect = np.eye(3)[y]
    assert discrimination_metrics(y, perfect)['gini_macro'] == 1
    assert discrimination_metrics(y, np.ones((6, 3)) / 3)['gini_macro'] == 0
    assert discrimination_metrics(y, (1 - perfect) / 2)['gini_macro'] == -1
    missing = discrimination_metrics([0, 0], [[.6, .3, .1], [.5, .3, .2]])
    assert np.isnan(missing['gini_macro'])
    assert np.isnan(missing['gini_home'])


def test_missing_test_class_does_not_become_a_zero_gini():
    frame = sample()
    frame.loc[frame.MatchDate.dt.year == 2025, 'target_label'] = 0
    scores, outcomes, dummy, _ = evaluate_individual_features(frame, ['signal'])
    assert np.isnan(scores.gini_macro.iloc[0])
    assert outcomes.dummy_gini.isna().all()
    assert np.isfinite(dummy['log_loss'])


def test_changing_test_cannot_change_dummy_or_fitted_logistic_coefficients(monkeypatch):
    from src import feature_diagnostics as module
    original = module.LogisticRegression.fit
    coefficients = []

    def record_fit(self, X, y, *args, **kwargs):
        result = original(self, X, y, *args, **kwargs)
        coefficients.append((self.coef_.copy(), self.intercept_.copy()))
        return result

    monkeypatch.setattr(module.LogisticRegression, 'fit', record_fit)
    frame = sample()
    first = evaluate_individual_features(frame, ['signal'])
    test = frame.MatchDate.dt.year == 2025
    frame.loc[test, 'signal'] = 10000
    frame.loc[test, 'target_label'] = np.arange(test.sum()) % 3
    second = evaluate_individual_features(frame, ['signal'])
    np.testing.assert_array_equal(first[1].dummy_probability, second[1].dummy_probability)
    np.testing.assert_allclose(coefficients[0][0], coefficients[1][0])
    np.testing.assert_allclose(coefficients[0][1], coefficients[1][1])


def test_missing_train_class_and_unknown_test_categories_remain_evaluable():
    frame = sample()
    train = frame.MatchDate.dt.year == 2020
    frame.loc[train & frame.target_label.eq(2), 'target_label'] = 0
    frame.loc[~train, 'category'] = 'new'
    scores, outcomes, _, _ = evaluate_individual_features(frame, ['category'])
    assert np.isfinite(scores.log_loss.iloc[0])
    assert outcomes.loc[outcomes.key.eq('away'), 'dummy_probability'].iloc[0] == 0
    assert 'saknas i train' in scores.status.iloc[0]


def test_catalog_covers_every_feature_with_explicit_logic_and_consistent_aliases():
    catalog = build_feature_catalog()
    assert set(catalog) == {'League', *ALL_FEATURE_COLUMNS}
    for entry in catalog.values():
        assert entry['description'] and entry['formula'] and entry['notes'] and entry['group']
    for alias, canonical in FEATURE_ALIASES.items():
        assert catalog[alias]['formula'] == catalog[canonical]['formula']
    assert 'före matchdagen' in catalog['home_position']['description']
    assert 'nollställs' in catalog['home_position']['notes']
