import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.model_experiments import (eligible_features, experiment_splits, generate_combinations,
                                    logistic_report, rank_combinations, run_combination_search,
                                    screen_training_features)


def data():
    rng = np.random.default_rng(5)
    n = 360
    y = np.arange(n) % 3
    frame = pd.DataFrame({'MatchDate': pd.date_range('2010-01-01', periods=n), 'target_label': y,
                          'signal': y + rng.normal(0, .1, n), 'constant': 0.})
    for i in range(8):
        frame[f'x{i}'] = y + rng.normal(0, .2 + i, n)
    return frame


def test_report_compares_same_dummy_and_model_across_disjoint_periods():
    frame = data()
    report = logistic_report(frame, ['signal', 'constant'])
    splits = report['splits'].set_index('partition')
    assert splits.loc['Train', 'end'] < splits.loc['Validering', 'start']
    assert splits.loc['Validering', 'end'] < splits.loc['Test', 'start']
    for partition, predictions in report['predictions'].items():
        assert len(predictions) == splits.loc[partition, 'rows']
        np.testing.assert_allclose(predictions[['prob_home', 'prob_draw', 'prob_away']].sum(axis=1), 1)
    outcomes = report['outcomes']
    np.testing.assert_allclose(outcomes.groupby('partition').model_probability.sum(), 1)
    for _, group in outcomes.groupby('outcome'):
        assert group.dummy_probability.nunique() == 1
    metrics = report['metrics'].set_index(['partition', 'model'])
    assert metrics.loc[('Test', 'LG'), 'gini_macro'] > .9
    assert metrics.loc[('Test', 'LG'), 'log_loss'] < metrics.loc[('Test', 'Dummy'), 'log_loss']


def test_test_labels_do_not_change_predictions_or_feature_screening():
    frame = data()
    columns = ['signal', 'constant', 'x0', 'x1']
    parts = experiment_splits(frame, columns)
    first_screen = screen_training_features(parts['Train'], columns)
    report = logistic_report(frame, columns)
    test_dates = parts['Test'].MatchDate
    frame.loc[frame.MatchDate.isin(test_dates), 'target_label'] = 0
    second_parts = experiment_splits(frame, columns)
    second_screen = screen_training_features(second_parts['Train'], columns)
    second_report = logistic_report(frame, columns)
    assert_frame_equal(first_screen, second_screen)
    for partition in ('Train', 'Test'):
        assert_frame_equal(report['predictions'][partition].drop(columns='target_label'),
                           second_report['predictions'][partition].drop(columns='target_label'))
    assert first_screen.fit_end.max() < first_screen.screen_start.min()
    assert first_screen.screen_end.max() <= parts['Train'].MatchDate.max()
    assert 'signal' in eligible_features(first_screen, .02)
    assert 'constant' not in eligible_features(first_screen, 0)


def test_combinations_are_unique_reproducible_and_obey_feature_limits():
    features = [f'x{i}' for i in range(10)]
    chosen = generate_combinations(features, 50, 5, 8, 42)
    assert len(chosen) == len(set(chosen)) == 50
    assert all(5 <= len(c) <= 8 for c in chosen)
    assert all(set(c) <= set(features) for c in chosen)
    assert chosen == generate_combinations(features[::-1], 50, 5, 8, 42)
    assert chosen != generate_combinations(features, 50, 5, 8, 43)
    assert generate_combinations(['a', 'b'], 50, 2, 2) == [('a', 'b')]
    # Alias and canonical name cannot occupy two slots in a model.
    assert generate_combinations(['home_form_points_last_5', 'home_points_last_5'], 50, 1, 1) == [('home_points_last_5',)]
    with pytest.raises(ValueError):
        generate_combinations(features, 50, 5, 11)
    with pytest.raises(ValueError):
        generate_combinations(features, 0, 5, 8)


def test_fifty_simulations_keep_same_rows_and_ranking_matches_explicit_metric():
    frame = data()
    progress = []
    results = run_combination_search(frame, [f'x{i}' for i in range(8)], 50, 3, 5,
                                     progress=lambda current, total: progress.append((current, total)))
    assert len(results) == 50
    assert results.n_features.between(3, 5).all()
    assert results.test_rows.nunique() == results.train_rows.nunique() == 1
    assert progress[-1] == (50, 50)
    assert rank_combinations(results, 'test', 'gini_macro').iloc[0].test_gini_macro == results.test_gini_macro.max()
    assert rank_combinations(results, 'validation', 'log_loss').iloc[0].validation_log_loss == results.validation_log_loss.min()


def test_undefined_and_unconverged_models_cannot_win():
    results = pd.DataFrame({'simulation': [1, 2, 3], 'status': ['OK', 'Ej konvergerad', 'OK'],
                            'test_gini_macro': [.2, .9, np.nan], 'test_log_loss': [1., .1, 2.]})
    assert rank_combinations(results).simulation.tolist() == [1]
    assert rank_combinations(results, metric='log_loss').simulation.tolist() == [1, 3]
    screening = pd.DataFrame({'feature': ['ok', 'bad', 'missing'], 'screen_gini': [.1, .9, np.nan],
                              'screen_status': ['OK', 'Ej konvergerad', 'OK']})
    assert eligible_features(screening) == ['ok']
