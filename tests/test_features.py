import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.feature_engineering import ALL_FEATURE_COLUMNS, FEATURE_ALIASES, engineer_features
from src.count_ratings import CountRatings


def match(date, home='A', away='B', hg=1, ag=1, season='2023/24', league='L', **stats):
    row = dict(MatchDate=date, Season=season, League=league, HomeTeam=home, AwayTeam=away,
               FullTimeHomeGoals=hg, FullTimeAwayGoals=ag,
               HomeShots=10, AwayShots=10, HomeShotsOnTarget=4, AwayShotsOnTarget=4,
               HomeCorners=5, AwayCorners=5)
    row.update(stats)
    return row


def test_elo_draw_zero_sum_and_home_advantage():
    rows = [match('2023-08-01'), match('2023-08-08')]
    f = engineer_features(pd.DataFrame(rows), elo_home_advantage=0)
    assert f.loc[1, 'home_elo'] == f.loc[1, 'away_elo'] == 1500
    f = engineer_features(pd.DataFrame(rows))
    assert f.loc[0, 'home_elo_expected'] > .5
    assert f.loc[1, 'home_elo'] < 1500
    assert f.loc[1, 'home_elo'] + f.loc[1, 'away_elo'] == pytest.approx(3000)
    assert f.loc[0, 'home_elo_expected'] + f.loc[0, 'away_elo_expected'] == 1


def test_per_match_momentum_and_relative_goal_strength():
    rows = [match(day) for day in pd.date_range('2023-08-01', periods=11, freq='7D')]
    f = engineer_features(pd.DataFrame(rows))
    for column in ('home_shot_momentum_l3_l10', 'home_sot_momentum_l3_l10', 'home_corner_momentum_l3_l10'):
        assert f.loc[10, column] == 0
    assert f.loc[10, 'home_attack_strength_l10'] == 1
    assert f.loc[10, 'away_defensive_strength_l10'] == 1
    assert f.loc[10, 'home_shot_share_l10'] == .5
    assert f.loc[10, 'home_goal_diff_std_l10'] == 0
    assert f.loc[10, 'home_rest_days'] == 7
    assert f.loc[10, 'home_matches_last_14_days'] == 2
    assert f.loc[10, 'home_sot_diff_per_match_l10'] == 0


def test_season_reset_preserves_form_and_rating():
    f = engineer_features(pd.DataFrame([
        match('2023-08-01', hg=2, ag=0),
        match('2024-08-01', season='2024/25', hg=0, ag=0),
        match('2024-08-08', season='2024/25'),
    ]))
    assert f.loc[1, 'home_points_last_5'] == 3
    assert f.loc[1, 'home_elo'] > 1500
    assert f.loc[1, 'home_points'] == 0
    assert f.loc[2, 'home_points'] == 1
    assert f.loc[2, 'away_points'] == 1
    assert f.loc[2, 'home_position'] == 1


def test_explicit_season_overrides_calendar_and_fallback_is_configurable():
    rows = [match('2020-06-30', hg=2, ag=0, season='2019/20'),
            match('2020-07-08', season='2019/20')]
    f = engineer_features(pd.DataFrame(rows))
    assert f.loc[1, 'home_points'] == 3  # delayed season, not a July reset
    frame = pd.DataFrame(rows).drop(columns='Season')
    assert engineer_features(frame).loc[1, 'home_points'] == 0
    assert engineer_features(frame, season_start_month=1).loc[1, 'home_points'] == 3


def test_leagues_are_isolated_even_with_identical_team_names():
    f = engineer_features(pd.DataFrame([
        match('2023-08-01', hg=4, ag=0), match('2023-08-08', league='Other')]))
    assert f.loc[1, 'home_elo'] == 1500
    assert f.loc[1, 'home_history_matches'] == 0


def test_current_same_day_and_future_results_cannot_leak():
    rows = [match(day) for day in pd.date_range('2023-08-01', periods=28, freq='D')]
    rows += [match('2023-08-28', home='C', away='D'), match('2023-09-10')]
    original = engineer_features(pd.DataFrame(rows))
    changed = [r.copy() for r in rows]
    changed[-3]['FullTimeHomeGoals'] = 9
    changed[-3]['HomeShots'] = 100
    changed[-2]['FullTimeAwayGoals'] = 8
    changed[-1]['FullTimeHomeGoals'] = 7
    updated = engineer_features(pd.DataFrame(changed))
    before = original.MatchDate <= pd.Timestamp('2023-08-28')
    assert_frame_equal(original.loc[before, ALL_FEATURE_COLUMNS], updated.loc[before, ALL_FEATURE_COLUMNS])
    prefix = engineer_features(pd.DataFrame(rows[:-1]))
    assert_frame_equal(original.iloc[:-1][ALL_FEATURE_COLUMNS], prefix[ALL_FEATURE_COLUMNS])
    assert_frame_equal(original, engineer_features(pd.DataFrame(rows).sample(frac=1, random_state=4)))


def test_unplayed_matches_do_not_update_any_history():
    rows = [match('2023-08-01'), match('2023-08-08', hg=np.nan, ag=np.nan), match('2023-08-15')]
    f = engineer_features(pd.DataFrame(rows))
    assert pd.isna(f.loc[1, 'target_label'])
    assert f.loc[2, 'home_history_matches'] == 1
    assert f.loc[2, 'home_rest_days'] == 14
    assert f.loc[2, 'home_points'] == 1
    without_fixture = engineer_features(pd.DataFrame([rows[0], rows[2]]))
    np.testing.assert_allclose(f.loc[2, ALL_FEATURE_COLUMNS].to_numpy(dtype=float),
                               without_fixture.loc[1, ALL_FEATURE_COLUMNS].to_numpy(dtype=float), equal_nan=True)


def test_missing_statistics_are_not_zero_or_poisoned_forever():
    rows = [match('2023-08-01', HomeShots=np.nan), match('2023-08-08'), match('2023-08-15')]
    f = engineer_features(pd.DataFrame(rows))
    assert pd.isna(f.loc[1, 'home_shots_for_l5'])
    assert pd.isna(f.loc[1, 'home_shot_share_l10'])
    assert f.loc[2, 'home_shots_for_ewm'] == pytest.approx(10)
    assert f.loc[2, 'home_shot_observations_l10'] == 1
    assert f.loc[2, 'home_shot_share_l10'] == .5
    assert not np.isinf(f[ALL_FEATURE_COLUMNS].to_numpy(dtype=float)).any()


def test_ewm_and_elo_opponent_context_have_explicit_values():
    rows = [match('2023-08-01', hg=3, ag=0), match('2023-09-30', hg=0, ag=0), match('2023-10-01')]
    f = engineer_features(pd.DataFrame(rows), elo_home_advantage=0)
    assert f.loc[2, 'home_goals_for_ewm'] == pytest.approx(1)  # weights 1:2
    assert f.loc[2, 'home_opponent_elo_mean_l5'] == pytest.approx((1500 + 1484) / 2)
    assert f.loc[1, 'home_result_overperformance_l5'] == .5


def test_poisson_probabilities_and_aliases():
    f = engineer_features(pd.DataFrame([match('2023-08-01')]))
    assert f.loc[0, 'expected_home_goals'] == pytest.approx(1.5)
    assert f.loc[0, 'expected_away_goals'] == pytest.approx(1.2)
    assert f.loc[0, 'expected_total_goals'] == pytest.approx(2.7)
    assert f.loc[0, ['poisson_prob_home', 'poisson_prob_draw', 'poisson_prob_away']].sum() == pytest.approx(1)
    for alias, canonical in FEATURE_ALIASES.items():
        pd.testing.assert_series_equal(f[alias], f[canonical], check_names=False)


def test_count_model_learns_attack_and_predict_uses_opponent_defence():
    model = CountRatings(1.2)
    for date in pd.date_range('2023-01-01', periods=40):
        model.observe(date, 'A', 'B', 4, 0)
        model.observe(date, 'C', 'D', 1, 1)
    model.fit_before(pd.Timestamp('2023-03-01'))
    assert model.strengths('A')[0] > model.strengths('C')[0]
    assert model.strengths('B')[1] > model.strengths('D')[1]
    assert model.predict('A', 'B')[0] > model.predict('A', 'D')[0]
    assert model.strengths('Unseen') == (1, 1)
    assert all(np.isfinite(model.predict('Unseen', 'A')))


def test_count_model_fit_excludes_current_and_future_observations():
    base, polluted = CountRatings(1.2), CountRatings(1.2)
    for date in pd.date_range('2023-01-01', periods=30):
        base.observe(date, 'A', 'B', 1, 1)
        polluted.observe(date, 'A', 'B', 1, 1)
    polluted.observe(pd.Timestamp('2023-03-01'), 'A', 'B', 50, 0)
    polluted.observe(pd.Timestamp('2024-01-01'), 'A', 'B', 50, 0)
    for model in (base, polluted):
        model.fit_before(pd.Timestamp('2023-03-01'))
    assert base.predict('A', 'B') == polluted.predict('A', 'B')


@pytest.mark.parametrize('change', [{'MatchDate': 'bad'}, {'FullTimeResult': 'A'}, {'HomeTeam': 'B'}])
def test_invalid_identity_and_contradictory_results_raise(change):
    with pytest.raises(ValueError):
        engineer_features(pd.DataFrame([match('2023-08-01', **change)] if 'MatchDate' not in change
                                      else [{**match('2023-08-01'), **change}]))


def test_duplicate_match_rejected_and_empty_input_supported():
    row = match('2023-08-01')
    with pytest.raises(ValueError, match='Duplicate'):
        engineer_features(pd.DataFrame([row, row]))
    assert engineer_features(pd.DataFrame([row]).iloc[:0]).empty
