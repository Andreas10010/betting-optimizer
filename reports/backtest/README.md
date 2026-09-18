# Betting backtest — 2026-09-16

Does the model's disagreement with the bookmaker make money? Log loss and accuracy
cannot answer that, because profit depends on price. This run settles simulated bets
at real closing odds.

```bash
PYTHONPATH=. python -m src.backtest
```

Data: football-data.co.uk via `src/football_data.py`, five leagues, 46,903 matches.
Model: `team_model.fit_best_logistic` (feature set `form_strength_model`, `C=0.03`),
refit before each predicted season on that season's predecessors only. Team-level
predictions are pooled into one H/D/A distribution per match. 22,559 matches had both
a prediction and Pinnacle closing odds.

## Forecast quality, identical matches

| | Log loss | Accuracy |
| --- | ---: | ---: |
| Model | 0.9815 | 52.9% |
| Pinnacle closing | 0.9622 | 54.1% |
| Constant training priors | 1.0695 | — |

The model closes about 82% of the gap between constant priors and the closing line,
and is behind it on every outcome (macro Gini 0.332 vs 0.369).

## Flat 1-unit bets at Pinnacle closing

| Edge threshold | Bets | Win rate | ROI | 95% CI |
| ---: | ---: | ---: | ---: | :--- |
| 0.00 | 29,208 | 28.5% | −6.57% | [−8.61%, −4.56%] |
| 0.05 | 21,239 | 26.3% | −6.89% | [−9.37%, −4.39%] |
| 0.10 | 15,552 | 24.1% | −8.91% | [−11.97%, −5.81%] |
| 0.20 | 8,492 | 20.5% | −10.75% | [−15.23%, −6.23%] |

Returns fall as the threshold rises. Expected value is computed from the model's own
probability, so when the model is the weaker forecaster a max-EV filter selects the
matches where it is most wrong.

## Controls

| | ROI |
| --- | ---: |
| One random outcome per match | −2.47% |
| Betting every outcome of every match | −3.68% |
| Model, edge > 0.05 | −6.89% |

Random betting loses the bookmaker's margin. The model loses 4.4 points more than that,
so its selection is worse than no model at all. At best-available prices across ~20 books
(`MaxC`), where random betting returns +1.00%, the model still returns −2.03%.

All 13 predicted seasons lose money (`by_season.csv`); all five leagues lose. Losses
concentrate on market longshots: bets at market probability below 15% return −15.66%.

## No-lookahead check

Features are recomputed for every match from all matches played before its date,
including earlier matches in the same season: that information exists when the bet is
placed. Model coefficients are a separate matter and are fitted once per season, on
earlier seasons only, so the model never trains on the season it is scored against.
Refitting during the season would only improve it, so this is conservative.

`tests/test_no_lookahead.py` enforces the first half of that: deleting every later match
must leave earlier features bit-identical. It includes a planted leak, so the guard is
known to be able to fail.

## Scope

This measures one model against one market. It does not show that no model can beat
this market, nor that this model would lose on other markets or leagues. Any leakage
remaining in the feature pipeline would flatter the model, so the negative result is
conservative with respect to that failure mode.
