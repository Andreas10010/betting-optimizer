import numpy as np
import pandas as pd

from src import backtest


def test_devig_removes_the_margin():
    # Inverses are 0.5556/0.2778/0.2778, summing to 1.111: an 11.1% overround.
    odds = pd.DataFrame({"H": [1.8], "D": [3.6], "A": [3.6]})
    probabilities = backtest.devig(odds)
    np.testing.assert_allclose(probabilities[0], [0.5, 0.25, 0.25])
    assert abs(probabilities.sum() - 1.0) < 1e-12


def test_devig_is_a_no_op_on_a_book_with_no_margin():
    odds = pd.DataFrame({"H": [2.0], "D": [4.0], "A": [4.0]})  # Inverses sum to 1.
    np.testing.assert_allclose(backtest.devig(odds)[0], [0.5, 0.25, 0.25])


def test_match_probabilities_merges_both_team_rows_into_one_distribution():
    team_frame = pd.DataFrame({"MatchId": ["m1", "m1"], "IsHome": [1, 0]})
    # CLASS_ORDER is [Förlust, Oavgjort, Vinst]. The home row says 60% win; the
    # away row says 50% loss, which is the same 50% home win seen from the other side.
    probabilities = np.array([[0.2, 0.2, 0.6],
                              [0.5, 0.2, 0.3]])
    out = backtest.match_probabilities(team_frame, probabilities)
    assert len(out) == 1
    np.testing.assert_allclose(out.loc["m1"].to_numpy(), [0.55, 0.20, 0.25])
    assert abs(out.loc["m1"].sum() - 1.0) < 1e-12


def test_simulate_settles_winners_at_the_price_paid_and_losers_at_minus_one():
    joined = pd.DataFrame({
        "H": [0.60], "D": [0.20], "A": [0.20],
        "market_H": [0.50], "market_D": [0.25], "market_A": [0.25],
        "price_H": [2.0], "price_D": [4.0], "price_A": [4.0],
        "result": ["H"], "League": ["Premier League"], "Season": ["2020/21"],
    })
    bets = backtest.simulate(joined, edge_threshold=0.05)
    # Only H clears: 0.60 * 2.0 - 1 = +0.20. D and A are 4.0 * 0.20 - 1 = -0.20.
    assert list(bets.outcome) == ["H"]
    assert bets.profit.iloc[0] == 1.0
    assert bool(bets.won.iloc[0])


def test_a_bet_that_loses_costs_exactly_the_stake():
    joined = pd.DataFrame({
        "H": [0.60], "D": [0.20], "A": [0.20],
        "market_H": [0.50], "market_D": [0.25], "market_A": [0.25],
        "price_H": [2.0], "price_D": [4.0], "price_A": [4.0],
        "result": ["A"], "League": ["Serie A"], "Season": ["2020/21"],
    })
    bets = backtest.simulate(joined, edge_threshold=0.05)
    assert bets.profit.iloc[0] == -1.0


def test_no_bet_is_placed_when_nothing_clears_the_threshold():
    joined = pd.DataFrame({
        "H": [0.50], "D": [0.25], "A": [0.25],
        "market_H": [0.50], "market_D": [0.25], "market_A": [0.25],
        "price_H": [1.9], "price_D": [3.8], "price_A": [3.8],
        "result": ["H"], "League": ["Ligue 1"], "Season": ["2020/21"],
    })
    assert backtest.simulate(joined, edge_threshold=0.05).empty


def test_walk_forward_never_trains_on_the_season_it_predicts(monkeypatch):
    seen = []

    def spy(train_df, *args, **kwargs):
        seen.append(sorted(train_df["Season"].astype(str).unique()))
        model = type("M", (), {})()
        model.named_steps = {"model": type("C", (), {"classes_": np.array(backtest.CLASS_ORDER)})()}
        model.predict_proba = lambda frame: np.tile([0.3, 0.3, 0.4], (len(frame), 1))
        return model

    monkeypatch.setattr(backtest, "fit_best_logistic", spy)
    monkeypatch.setattr(backtest, "model_input_columns", lambda: ["Venue"])
    team_data = pd.DataFrame({
        "Season": ["2020/21", "2020/21", "2021/22", "2021/22", "2022/23", "2022/23"],
        "MatchId": ["a", "a", "b", "b", "c", "c"],
        "IsHome": [1, 0, 1, 0, 1, 0],
        "Venue": ["Hemma", "Borta"] * 3,
        "Target": ["Vinst", "Förlust"] * 3,
    })
    backtest.walk_forward_predictions(team_data, "2021/22", verbose=False)
    assert seen == [["2020/21"], ["2020/21", "2021/22"]]
