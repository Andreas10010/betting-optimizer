"""Walk-forward betting backtest: does the model's disagreement with the market pay?

Accuracy and log loss say whether the model forecasts well. They cannot say
whether it makes money, because that depends on price. This module refits the
model season by season, converts its team-level output into one H/D/A
distribution per match, and settles simulated bets at real bookmaker prices.

Every season is predicted by a model fitted only on earlier seasons, so no bet
uses information from its own season or later.

    PYTHONPATH=. python -m src.backtest --first-test-season 2013
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_engineering import engineer_features
from .football_data import load_matches
from .team_dataset import build_team_match_dataset
from .team_model import CLASS_ORDER, fit_best_logistic, model_input_columns

OUTCOMES = ["H", "D", "A"]
# Pinnacle closing: the sharpest line in the data and the cheapest margin, so
# the hardest honest benchmark. Opening prices are what you could bet early.
CLOSING_ODDS = ["PSCH", "PSCD", "PSCA"]
OPENING_ODDS = ["PSH", "PSD", "PSA"]


def devig(odds: pd.DataFrame) -> np.ndarray:
    """Implied probabilities with the bookmaker's margin divided out proportionally."""
    inverse = 1.0 / odds.to_numpy(dtype=float)
    return inverse / inverse.sum(axis=1, keepdims=True)


def match_probabilities(team_frame: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    """Fold the two team-level rows of a match into one H/D/A distribution.

    The model scores each team separately, so a match yields two opinions that
    need not agree. Averaging the home team's view with the mirrored away view
    uses both, then renormalises.
    """
    loss, draw, win = (probabilities[:, CLASS_ORDER.index(label)]
                       for label in ("Förlust", "Oavgjort", "Vinst"))
    frame = pd.DataFrame({
        "MatchId": team_frame["MatchId"].to_numpy(),
        "IsHome": team_frame["IsHome"].to_numpy(),
        # Rewrite every row as (home win, draw, away win).
        "H": np.where(team_frame["IsHome"].to_numpy() == 1, win, loss),
        "D": draw,
        "A": np.where(team_frame["IsHome"].to_numpy() == 1, loss, win),
    })
    pooled = frame.groupby("MatchId")[OUTCOMES].mean()
    return pooled.div(pooled.sum(axis=1), axis=0)


def walk_forward_predictions(team_data: pd.DataFrame, first_test_season: str,
                             verbose: bool = True) -> pd.DataFrame:
    """Predict each season with a model fitted only on the seasons before it."""
    data = team_data.dropna(subset=["Target"])
    data = data.loc[data["Target"].isin(CLASS_ORDER)]
    seasons = sorted(data["Season"].astype(str).unique())
    columns = model_input_columns()
    predictions = []
    for season in [s for s in seasons if s >= first_test_season]:
        season_labels = data["Season"].astype(str)
        train = data.loc[season_labels.lt(season)]
        test = data.loc[season_labels.eq(season)]
        if train.empty or test.empty or train["Target"].nunique() < 2:
            continue
        model = fit_best_logistic(train)
        fitted = list(model.named_steps["model"].classes_)
        raw = model.predict_proba(test[columns])
        ordered = np.column_stack([raw[:, fitted.index(label)] for label in CLASS_ORDER])
        frame = match_probabilities(test, ordered)
        frame["Season"] = season
        predictions.append(frame)
        if verbose:
            print(f"  {season}: trained on {len(train):,} team-rows, predicted {len(frame):,} matches")
    if not predictions:
        raise ValueError("No season could be both trained and tested")
    return pd.concat(predictions)


def build_market(raw: pd.DataFrame, odds_columns=CLOSING_ODDS) -> pd.DataFrame:
    """Devigged market probabilities and the prices bets are settled at."""
    from .team_dataset import match_id

    usable = raw.dropna(subset=odds_columns + ["FullTimeResult"]).copy()
    usable = usable.loc[(usable[odds_columns] > 1).all(axis=1)]
    market = pd.DataFrame(devig(usable[odds_columns]), columns=OUTCOMES,
                          index=match_id(usable)).add_prefix("market_")
    prices = pd.DataFrame(usable[odds_columns].to_numpy(dtype=float), columns=OUTCOMES,
                          index=market.index).add_prefix("price_")
    result = pd.concat([market, prices], axis=1)
    result["result"] = usable["FullTimeResult"].to_numpy()
    result["League"] = usable["League"].to_numpy()
    return result.loc[~result.index.duplicated()]


def simulate(joined: pd.DataFrame, edge_threshold: float) -> pd.DataFrame:
    """Flat 1-unit bets on every outcome whose expected value clears the threshold.

    Expected value uses the price actually paid, so this asks "is the model's
    probability high enough to make this price worth taking", not merely "does
    the model disagree with the market".
    """
    bets = []
    for outcome in OUTCOMES:
        expected = joined[outcome] * joined[f"price_{outcome}"] - 1.0
        chosen = joined.loc[expected > edge_threshold]
        if chosen.empty:
            continue
        won = chosen["result"].eq(outcome)
        bets.append(pd.DataFrame({
            "Season": chosen["Season"], "League": chosen["League"], "outcome": outcome,
            "price": chosen[f"price_{outcome}"],
            "model_prob": chosen[outcome], "market_prob": chosen[f"market_{outcome}"],
            "expected_value": expected.loc[chosen.index],
            "profit": np.where(won, chosen[f"price_{outcome}"] - 1.0, -1.0),
            "won": won,
        }))
    return pd.concat(bets) if bets else pd.DataFrame(
        columns=["Season", "League", "outcome", "price", "model_prob",
                 "market_prob", "expected_value", "profit", "won"])


def bootstrap_roi(profit: np.ndarray, draws: int = 10_000, seed: int = 0):
    """Percentile interval for ROI. Flat bets are independent, so resample them."""
    if len(profit) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    sample = rng.choice(profit, size=(draws, len(profit)), replace=True).mean(axis=1)
    return tuple(np.percentile(sample, [2.5, 97.5]))


def run(first_test_season: str = "2013/14", divisions=None, verbose: bool = True):
    raw = load_matches(divisions=divisions)
    if verbose:
        print(f"Loaded {len(raw):,} matches. Engineering features...")
    engineered = engineer_features(raw)
    team_data = build_team_match_dataset(raw, engineered)
    if verbose:
        print(f"Team-level rows: {len(team_data):,}. Walk-forward refit per season:")
    predictions = walk_forward_predictions(team_data, first_test_season, verbose)
    market = build_market(raw)
    joined = predictions.join(market, how="inner")
    if verbose:
        print(f"\nMatches with both a prediction and Pinnacle closing odds: {len(joined):,}")
    return joined


def summarize(joined: pd.DataFrame, thresholds=(0.0, 0.05, 0.10, 0.20)) -> pd.DataFrame:
    """ROI per edge threshold, with the no-skill control it has to beat."""
    rows = []
    for threshold in thresholds:
        bets = simulate(joined, threshold)
        if bets.empty:
            continue
        low, high = bootstrap_roi(bets["profit"].to_numpy())
        rows.append({
            "edge_threshold": threshold, "bets": len(bets),
            "win_rate": float(bets["won"].mean()), "roi": float(bets["profit"].mean()),
            "roi_ci_low": low, "roi_ci_high": high,
            "profit_units": float(bets["profit"].sum()),
        })
    return pd.DataFrame(rows)


def random_bet_control(joined: pd.DataFrame, seed: int = 0) -> float:
    """ROI of betting one outcome per match at random: the bookmaker's margin.

    Any strategy that does not beat this is worse than having no model at all.
    """
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(OUTCOMES), len(joined))
    chosen = np.array(OUTCOMES)[picks]
    prices = joined[[f"price_{o}" for o in OUTCOMES]].to_numpy(dtype=float)
    won = chosen == joined["result"].to_numpy()
    return float(np.where(won, prices[np.arange(len(joined)), picks] - 1.0, -1.0).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-test-season", default="2013/14",
                        help="First season to predict; earlier ones are training data only")
    parser.add_argument("--divisions", nargs="+", default=None,
                        help="football-data division codes, e.g. E0 D1 SP1 I1 F1")
    parser.add_argument("--output", type=Path, default=Path("reports/backtest"))
    args = parser.parse_args()

    joined = run(first_test_season=args.first_test_season, divisions=args.divisions)
    summary = summarize(joined)
    control = random_bet_control(joined)

    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "summary.csv", index=False)
    bets = simulate(joined, 0.05)
    bets.groupby("Season").agg(bets=("profit", "size"), roi=("profit", "mean"),
                               profit_units=("profit", "sum")).to_csv(args.output / "by_season.csv")

    print(f"\nFlat 1-unit bets at Pinnacle closing prices, {len(joined):,} matches.")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    print(f"\nNo-skill control (one random outcome per match): {100 * control:+.2f}% ROI")
    print("A strategy must beat that control, not merely lose less than 100%.")
    print(f"Saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
