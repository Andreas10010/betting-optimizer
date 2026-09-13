import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_frame_equal, assert_series_equal

from src.feature_engineering import engineer_features
from src.team_dataset import build_team_match_dataset
from src.team_model import (CLASS_ORDER, CorrelationPruner, add_predictions, class_shares,
                            evaluate_predictions, explain_row, fit_best_logistic, logistic_pipeline,
                            model_input_columns, numeric_feature_columns, probabilities_in_order,
                            save_trained_model, load_trained_model, split_train_test,
                            train_all_history_model)


def match(date, home='A', away='B', hg=1, ag=1, season='2022/23', league='Premier League', **stats):
    if hg > ag:
        result = 'H'
    elif hg < ag:
        result = 'A'
    else:
        result = 'D'
    row = dict(MatchDate=date, Season=season, League=league, HomeTeam=home, AwayTeam=away,
               FullTimeHomeGoals=hg, FullTimeAwayGoals=ag, FullTimeResult=result,
               HomeShots=10, AwayShots=10, HomeShotsOnTarget=4, AwayShotsOnTarget=4,
               HomeCorners=5, AwayCorners=5, HomeYellowCards=1, AwayYellowCards=1,
               HomeRedCards=0, AwayRedCards=0)
    row.update(stats)
    return row


def mini_league():
    rows = [
        match('2021-08-14', 'A', 'B', 1, 0, season='2021/22'),
        match('2022-08-01', 'A', 'B', 2, 0),
        match('2022-08-08', 'C', 'D', 1, 1),
        match('2022-08-15', 'A', 'C', 3, 1),
        match('2022-08-22', 'B', 'D', 0, 2),
        match('2022-08-29', 'A', 'D', 1, 0),
        match('2023-08-12', 'A', 'B', 0, 1, season='2023/24'),
        match('2023-08-19', 'D', 'C', 2, 0, season='2023/24'),
        match('2023-08-26', 'B', 'C', 2, 2, season='2023/24'),
    ]
    raw = pd.DataFrame(rows)
    engineered = engineer_features(raw)
    return raw, engineered, build_team_match_dataset(raw, engineered)


def test_team_dataset_has_two_rows_per_match_and_team_targets():
    raw, _, team = mini_league()
    assert len(team) == 2 * len(raw)
    first = team[(team.Team == 'A') & (team.MatchDate == pd.Timestamp('2021-08-14'))].iloc[0]
    assert first.Opponent == 'B'
    assert first.Venue == 'Hemma'
    assert first.Target == 'Vinst'
    away = team[(team.Team == 'B') & (team.MatchDate == pd.Timestamp('2022-08-01'))].iloc[0]
    assert away.Target == 'Förlust'
    assert away.Venue == 'Borta'


def test_rolling_features_use_only_prior_matches_and_home_scope():
    _, _, team = mini_league()
    opening = team[(team.Team == 'A') & (team.MatchDate == pd.Timestamp('2021-08-14'))].iloc[0]
    assert opening.history_matches_total == 0
    assert pd.isna(opening.ppg_total_l5)
    second = team[(team.Team == 'A') & (team.MatchDate == pd.Timestamp('2022-08-01'))].iloc[0]
    assert second.history_matches_total == 1
    assert second.points_total_l5 == 3
    assert second.goal_diff_total_l5 == 1
    assert second.ppg_home_l5 == 3
    assert pd.isna(second.ppg_away_l5)
    later_home = team[(team.Team == 'A') & (team.MatchDate == pd.Timestamp('2022-08-29'))].iloc[0]
    assert later_home.points_home_l5 == 9
    assert later_home.history_matches_home == 3


def test_relative_strength_features_and_season_split():
    _, _, team = mini_league()
    row = team.iloc[-1]
    assert row.position_advantage == row.opponent_position - row.team_position
    assert row.elo_difference == pytest.approx(row.team_elo - row.opponent_elo)
    assert row.expected_goal_difference == pytest.approx(row.expected_team_goals - row.expected_opponent_goals)
    train, test = split_train_test(team)
    assert set(train.Season) == {'2021/22', '2022/23'}
    assert set(test.Season) == {'2023/24'}
    assert train.MatchDate.max() < test.MatchDate.min()


def test_correlation_pruner_keeps_first_of_highly_correlated_pair():
    values = np.array([[1., 10., 1.], [2., 20., 2.], [3., 30., 0.], [4., 40., 4.]])
    pruner = CorrelationPruner(threshold=0.90).fit(values)
    assert list(pruner.kept_indices_) == [0, 2]
    np.testing.assert_array_equal(pruner.transform(values), values[:, [0, 2]])
    np.testing.assert_array_equal(pruner.get_feature_names_out(['a', 'b', 'c']), np.array(['a', 'c'], dtype=object))


def logistic_frame(n=120, seed=0):
    rng = np.random.default_rng(seed)
    elo = rng.normal(0, 1, n)
    venue = np.where(rng.random(n) > 0.5, 'Hemma', 'Borta')
    score = elo + 0.6 * (venue == 'Hemma')
    target = np.where(score > 0.4, 'Vinst', np.where(score < -0.4, 'Förlust', 'Oavgjort'))
    frame = pd.DataFrame({
        'Team': np.where(rng.random(n) > 0.5, 'A', 'B'),
        'Opponent': 'C',
        'Venue': venue,
        'Target': target,
        'Season': ['2022/23'] * (n // 2) + ['2023/24'] * (n - n // 2),
        'MatchDate': pd.date_range('2022-08-01', periods=n, freq='D'),
        'MatchId': [f'm{i}' for i in range(n)],
    })
    for column in numeric_feature_columns():
        noise = rng.normal(0, 0.3, n)
        if column in ('elo_difference', 'position_advantage', 'expected_goal_difference', 'team_elo_expected'):
            frame[column] = elo + noise
        elif column == 'poisson_prob_win':
            frame[column] = 1 / (1 + np.exp(-score))
        elif column == 'poisson_prob_loss':
            frame[column] = 1 / (1 + np.exp(score))
        else:
            frame[column] = rng.normal(0, 1, n)
    return frame


def test_best_logistic_predicts_three_classes_and_explanations_match_logits():
    frame = logistic_frame()
    train, test = split_train_test(frame)
    model = fit_best_logistic(train)
    predicted = add_predictions(model, test)
    assert set(predicted.Predicted) <= set(CLASS_ORDER)
    np.testing.assert_allclose(
        predicted[['ProbabilityLoss', 'ProbabilityDraw', 'ProbabilityWin']].sum(axis=1), 1, atol=1e-9
    )
    metrics = evaluate_predictions(predicted)
    assert metrics['rows'] == len(test)
    assert metrics['accuracy'] > metrics['dummy_accuracy']
    row = predicted.iloc[0]
    explanation = explain_row(model, row)
    contributions = explanation['contributions']
    reconstructed = explanation['intercept'] + contributions['contribution'].sum()
    assert reconstructed == pytest.approx(explanation['logit'])
    transformed = model.named_steps['preprocessing'].transform(pd.DataFrame([row])[model_input_columns()])
    classes = list(model.named_steps['model'].classes_)
    class_index = classes.index(explanation['outcome'])
    expected_logit = (
        model.named_steps['model'].intercept_[class_index]
        + transformed.ravel() @ model.named_steps['model'].coef_[class_index]
    )
    assert explanation['logit'] == pytest.approx(float(expected_logit))
    assert explanation['predicted'] in CLASS_ORDER
    pipe = logistic_pipeline()
    assert pipe.named_steps['model'].C == 0.03


def test_all_history_split_and_saved_model_roundtrip(tmp_path):
    frame = logistic_frame()
    frame.loc[frame.index[:20], 'Season'] = '2021/22'
    train, test = split_train_test(frame)
    assert '2021/22' in set(train.Season)
    assert '2022/23' in set(train.Season)
    assert set(test.Season) == {'2023/24'}
    bundle = train_all_history_model(frame)
    path = tmp_path / 'premier_league_lg_all_history.joblib'
    save_trained_model(path, bundle)
    loaded = load_trained_model(path)
    np.testing.assert_allclose(
        add_predictions(bundle['model'], test)[['ProbabilityWin']].to_numpy(),
        add_predictions(loaded['model'], test)[['ProbabilityWin']].to_numpy(),
    )
    assert loaded['config']['feature_set'] == 'form_strength_model'
    assert loaded['train_end'] == '2022/23'


def _feature_cols():
    return model_input_columns() + ['history_matches_total', 'team_elo', 'opponent_elo',
                                    'team_position', 'poisson_prob_win']


def test_current_and_future_results_do_not_enter_pre_match_features():
    raw, _, team = mini_league()
    polluted = raw.copy()
    current = polluted.MatchDate.eq('2022-08-15') & polluted.HomeTeam.eq('A')
    future = polluted.MatchDate.eq('2023-08-12')
    polluted.loc[current, ['FullTimeHomeGoals', 'FullTimeAwayGoals', 'HomeShots', 'HomeShotsOnTarget']] = [0, 7, 99, 20]
    polluted.loc[current, 'FullTimeResult'] = 'A'
    polluted.loc[future, ['FullTimeHomeGoals', 'FullTimeAwayGoals']] = [9, 0]
    polluted.loc[future, 'FullTimeResult'] = 'H'
    polluted_team = build_team_match_dataset(polluted, engineer_features(polluted))
    date = pd.Timestamp('2022-08-15')
    original_row = team[(team.Team == 'A') & (team.MatchDate == date)].iloc[0]
    polluted_row = polluted_team[(polluted_team.Team == 'A') & (polluted_team.MatchDate == date)].iloc[0]
    assert original_row.Target == 'Vinst'
    assert polluted_row.Target == 'Förlust'
    for column in _feature_cols():
        left, right = original_row[column], polluted_row[column]
        if pd.isna(left) and pd.isna(right):
            continue
        assert left == pytest.approx(right)
    before = team.MatchDate < date
    assert_frame_equal(
        team.loc[before, _feature_cols()].reset_index(drop=True),
        polluted_team.loc[polluted_team.MatchDate < date, _feature_cols()].reset_index(drop=True),
    )


def test_test_labels_and_match_ids_cannot_leak_into_training():
    _, _, team = mini_league()
    train, test = split_train_test(team)
    assert set(train.MatchId).isdisjoint(set(test.MatchId))
    assert train.Season.max() < test.Season.min() or train.MatchDate.max() < test.MatchDate.min()
    assert not set(model_input_columns()) & {'Target', 'Predicted', 'Correct', 'MatchId', 'Team'}
    assert not any(column.startswith('_') for column in team.columns)
    model = fit_best_logistic(train)
    original = probabilities_in_order(model, test)
    leaked = team.copy()
    leaked.loc[leaked.Season.eq('2023/24'), 'Target'] = 'Förlust'
    train2, test2 = split_train_test(leaked)
    model2 = fit_best_logistic(train2)
    np.testing.assert_allclose(model.named_steps['model'].coef_, model2.named_steps['model'].coef_)
    np.testing.assert_allclose(original, probabilities_in_order(model2, test2))
    assert_frame_equal(test[model_input_columns()].reset_index(drop=True),
                       test2[model_input_columns()].reset_index(drop=True))
    bundle = train_all_history_model(team)
    np.testing.assert_allclose(bundle['metrics']['dummy_priors'], class_shares(train['Target']))
    assert not np.allclose(bundle['metrics']['dummy_priors'], class_shares(test['Target']))


def test_same_day_other_fixture_does_not_change_a_teams_features():
    rows = [
        match('2022-08-01', 'A', 'B', 2, 0),
        match('2022-08-15', 'A', 'C', 1, 0),
        match('2022-08-15', 'B', 'D', 0, 1),
    ]
    raw = pd.DataFrame(rows)
    team = build_team_match_dataset(raw, engineer_features(raw))
    polluted = raw.copy()
    other = polluted.HomeTeam.eq('B') & polluted.MatchDate.eq('2022-08-15')
    polluted.loc[other, ['FullTimeHomeGoals', 'FullTimeAwayGoals', 'HomeShots']] = [8, 0, 40]
    polluted.loc[other, 'FullTimeResult'] = 'H'
    polluted_team = build_team_match_dataset(polluted, engineer_features(polluted))
    date = pd.Timestamp('2022-08-15')
    original = team[(team.Team == 'A') & (team.MatchDate == date)].iloc[0]
    updated = polluted_team[(polluted_team.Team == 'A') & (polluted_team.MatchDate == date)].iloc[0]
    assert_series_equal(original[_feature_cols()], updated[_feature_cols()], check_names=False)
