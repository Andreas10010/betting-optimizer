import math

import pytest

from src.model_lab import FAIR_ODDS_PROB_TICKS, fair_decimal_odds, probability_figure
from src.team_model import CLASS_ORDER


def test_fair_decimal_odds_are_reciprocal_of_probability():
    assert fair_decimal_odds(1 / 3) == pytest.approx(3.0)
    assert fair_decimal_odds(0.5) == pytest.approx(2.0)
    assert fair_decimal_odds(1.0) == pytest.approx(1.0)
    assert math.isnan(fair_decimal_odds(0))
    assert math.isnan(fair_decimal_odds(-0.1))
    assert math.isnan(fair_decimal_odds(None))


def test_probability_figure_adds_right_axis_with_fair_odds():
    chart = probability_figure({"Förlust": 0.25, "Oavgjort": 0.25, "Vinst": 0.5})
    spec = chart.to_dict()
    dataset = next(iter(spec["datasets"].values()))
    by_outcome = {row["Utfall"]: row for row in dataset}
    assert by_outcome["Vinst"]["Odds"] == pytest.approx(2.0)
    assert by_outcome["Förlust"]["Odds"] == pytest.approx(4.0)
    axes = [layer["encoding"]["y"].get("axis") or {} for layer in spec["layer"]]
    titles = {axis.get("title") for axis in axes}
    assert "Sannolikhet (%)" in titles
    assert "Fair odds" in titles
    odds_axis = next(axis for axis in axes if axis.get("title") == "Fair odds")
    assert odds_axis["orient"] == "right"
    assert odds_axis["labelExpr"] == "format(100 / datum.value, '.2f')"
    assert odds_axis["values"] == pytest.approx(FAIR_ODDS_PROB_TICKS)
    assert spec["resolve"]["axis"]["y"] == "independent"
    assert [row["Utfall"] for row in dataset] == list(CLASS_ORDER)
