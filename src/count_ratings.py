"""Past-only, regularized, time-weighted Poisson attack/defence models.

Goals use an independent Poisson model (no Dixon–Coles low-score correction).
Shots and shots on target use the same mean model as count-based ratings.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class CountRatings:
    """log E[count] = intercept + home_advantage + attack + defence.

    Larger defence multipliers mean more conceded. Priors are fixed, never
    estimated on future rows. Refits use observations strictly before `date`.
    """

    def __init__(self, away_prior: float, home_ratio: float = 1.25,
                 half_life_days: float = 180.0, refit_days: int = 28):
        self.prior = np.log([away_prior, home_ratio])
        self.half_life_days = half_life_days
        self.refit_days = refit_days
        self.history: list[tuple] = []
        self.last_fit: pd.Timestamp | None = None
        self.global_params = self.prior.copy()
        self.ratings: dict[str, tuple[float, float]] = {}

    def observe(self, date, home, away, home_count, away_count):
        if np.isfinite(home_count) or np.isfinite(away_count):
            self.history.append((date, home, away, home_count, away_count))

    def fit_before(self, date: pd.Timestamp):
        if self.last_fit is not None and (date - self.last_fit).days < self.refit_days:
            return
        cutoff = date - pd.Timedelta(days=3 * 365)
        self.history = [row for row in self.history if row[0] >= cutoff]
        rows = [row for row in self.history if row[0] < date]
        if len(rows) < 20:
            return
        teams = sorted({team for row in rows for team in row[1:3]})
        lookup = {team: i for i, team in enumerate(teams)}
        n = len(teams)
        attackers, defenders, venues, counts, weights = [], [], [], [], []
        for played, home, away, hc, ac in rows:
            weight = 2 ** (-(date - played).days / self.half_life_days)
            for team, opponent, venue, count in ((home, away, 1, hc), (away, home, 0, ac)):
                if np.isfinite(count):
                    attackers.append(lookup[team])
                    defenders.append(lookup[opponent])
                    venues.append(venue)
                    counts.append(count)
                    weights.append(weight)
        a, d = np.array(attackers), np.array(defenders)
        venue, y, w = np.array(venues), np.array(counts), np.array(weights)
        initial = np.zeros(2 + 2 * n)
        initial[:2] = self.global_params
        for team, i in lookup.items():
            initial[2 + i], initial[2 + n + i] = self.ratings.get(team, (0.0, 0.0))

        def objective(params):
            eta = params[0] + params[1] * venue + params[2 + a] + params[2 + n + d]
            mu = np.exp(eta)
            # Ridge shrinkage identifies attack/defence and stabilizes new teams.
            penalty = params.copy()
            penalty[:2] -= self.prior
            precision = np.full_like(params, 5.0)
            precision[:2] = 1.0
            loss = np.sum(w * (mu - y * eta)) + .5 * np.sum(precision * penalty ** 2)
            residual = w * (mu - y)
            grad = precision * penalty
            grad[0] += residual.sum()
            grad[1] += residual @ venue
            grad[2:2 + n] += np.bincount(a, weights=residual, minlength=n)
            grad[2 + n:] += np.bincount(d, weights=residual, minlength=n)
            return loss, grad

        result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                          bounds=[(-4, 4), (-1, 1)] + [(-2, 2)] * (2 * n),
                          options={"maxiter": 300, "ftol": 1e-9})
        self.last_fit = date
        if not result.success or not np.isfinite(result.fun):
            warnings.warn(f"Count-rating fit failed at {date}: {result.message}; keeping prior fit.", RuntimeWarning)
            return
        self.global_params = result.x[:2]
        self.ratings = {team: (result.x[2 + i], result.x[2 + n + i]) for team, i in lookup.items()}

    def predict(self, home: str, away: str) -> tuple[float, float]:
        ha, hd = self.ratings.get(home, (0.0, 0.0))
        aa, ad = self.ratings.get(away, (0.0, 0.0))
        base, advantage = self.global_params
        return float(np.exp(base + advantage + ha + ad)), float(np.exp(base + aa + hd))

    def strengths(self, team: str) -> tuple[float, float]:
        attack, defence = self.ratings.get(team, (0.0, 0.0))
        return float(np.exp(attack)), float(np.exp(defence))
