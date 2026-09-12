# Betting Optimizer

Pre-match football features and Home / Draw / Away models, with a Streamlit dashboard and chronological feature-family comparisons.

## Run

```bash
python3 -m pip install -r requirements.txt
streamlit run src/dashboard.py
```

Run tests or reproduce the feature benchmark:

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest -q
python3 -m src.evaluate_features --data /path/to/matches.csv --output reports/feature_benchmark
```

Omit `--data` to download the public dataset configured in `src/data_loader.py`. `--rows 5000` selects the first 5,000 CSV rows, **not the latest matches**. The CLI also accepts `--groups baseline_form baseline_poisson baseline_shot_ratings` and `--folds 3`.

## Feature groups

| Group | Inputs |
| --- | --- |
| `baseline_form` | Recent form, goal difference and corrected Elo |
| `extended_stats` | Baseline-style form plus rolling match statistics |
| `corrected_legacy` | All original feature definitions, with mathematical fixes and alias deduplication |
| `baseline_context` | Baseline plus time-weighted statistics, rest, schedule strength, dominance and variability |
| `baseline_poisson` | Baseline plus predicted goals and independent-Poisson H/D/A probabilities |
| `baseline_shot_ratings` | Baseline plus opponent-adjusted shot/SOT ratings and predicted counts |
| `all_features` | All canonical features, including the new groups |

`FEATURE_GROUPS` in `src/modeling.py` is the exact model-input registry. The dashboard displays the selected columns. Existing aliases remain in the engineered dataframe for compatibility, but the model uses each definition only once. More features are candidates for evaluation, not a guarantee of better predictions.

## Temporal contract and missing data

Features for calendar date **D** use only completed matches dated **before D**. All matches on D are predicted before any result from D updates form, standings, Elo or count models. This deliberately excludes earlier kickoffs on the same day because the dataset does not provide reliable kickoff/result-availability times. Input row order does not change the output. Pass the historical dataset together with future fixtures; future fixtures need missing goal values and do not update state.

Team state is isolated by `(League, Team)`. Form and Elo carry across seasons within a league. Standings and league goal averages use `(League, Season)`. Explicit `Season` labels take precedence, including delayed seasons; if absent, a July boundary is used for the European leagues. Set `engineer_features(..., season_start_month=1)` for calendar-year seasons. Team name changes need upstream reconciliation.

Standings are reconstructed from observed results, with points / goal difference / goals scored / team name as tiebreakers. They do not include official point deductions, league-specific head-to-head rules, or clubs not yet observed. Season points begin at zero; PPG is missing until the first completed match.

Missing/negative/nonfinite/noninteger count statistics become missing, never zero. Historical totals require all observations in the available window; means use observed values. Shares use matches with both numerator and denominator statistics present. `history_matches`, `history_matches_l10` and `shot_observations_l10` expose some of the available evidence. Rest and congestion cover only matches recorded in this league dataset, not cups or international fixtures. Invalid dates, duplicate team/date fixtures and results contradicting complete goal scores raise errors.

Missing features do not discard labeled matches. Numeric median imputation and scaling are fitted on each **training fold only**, followed by logistic regression or random forest. Unknown categorical values are handled by the fitted encoder. A count feature with no observations in training is imputed to zero. Training requires at least two observed result classes; probability output is always ordered H/D/A.

## Mathematical definitions

For team `i`, let `R_k` be its last `min(k, available matches)` completed matches before D.

| Feature | Definition |
| --- | --- |
| `*_ppg_momentum_l3_l10` | mean(points in R3) − mean(points in R10) |
| `*_shot_momentum_l3_l10`, SOT and corners | mean(statistic in R3) − mean(statistic in R10); constant performance gives zero |
| `*_attack_strength_l10` | mean(goals scored in R10) / league goals per team-match this season |
| `*_defensive_strength_l10` | mean(goals conceded in R10) / league goals per team-match this season; larger means more conceded |
| `*_shot_share_l10`, `*_sot_share_l10` | sum(for) / sum(for + against), using paired observed matches |
| `*_sot_diff_per_match_l10` | mean(SOT for − SOT against), using paired observed matches |
| `*_opponent_elo_mean_l5` | mean of opponents' **pre-match** Elo ratings in R5 |
| `*_result_overperformance_l5` | mean(actual Elo score − pre-match Elo expectation) in R5 |
| `*_rest_days` | D − date of last recorded completed match |
| `*_matches_last_14_days` | number of recorded completed matches in [D − 14 days, D) |
| `*_goal_diff_std_l10` | sample standard deviation of goal difference in R10 (`ddof=1`; requires two matches) |
| `*_goals_for_ewm`, goals against, shots and SOT | sum(w × statistic) / sum(w), with w = 2^(−age_days / 60), restricted to 1,095 days and observed values |

The league goal baseline is `sum(home goals + away goals) / (2 × number of completed league matches this season)`. Thus relative strength 1 means the league's per-team average. Recent form can span seasons even though this denominator resets. A zero or unavailable denominator yields a missing feature.

Elo starts at 1500 and uses:

```text
E_home = 1 / (1 + 10^((R_away − R_home − H) / 400))
S_home = 1 for home win, 0.5 for draw, 0 for away win
change = 32 × (S_home − E_home)
R_home += change
R_away -= change
```

`H = 60` by default, configurable as `elo_home_advantage`. `home_adjusted_elo_diff = R_home + H − R_away`. Elo expectation is an expected result score, **not** the probability of a home win when draws are possible. Home and away updates sum to zero.

### Goal and shot models

`src/count_ratings.py` fits one count model per league and statistic: goals, shots, and shots on target. For team i against j:

```text
log(lambda_home) = intercept + home_advantage + attack_i + defence_j
log(lambda_away) = intercept + attack_j + defence_i
```

The objective is the sum of time-weighted Poisson negative log likelihoods, excluding the parameter-independent factorial term, plus an L2 penalty. Calendar-time weights are `2^(−age_days / 180)`. Training uses at most 1,095 days of past observations, and refits occur on the first match date at least 28 days after the last fit, once 20 observed historical matches exist. Parameters stay fixed between refits. All fixtures on the refit date are excluded from fitting.

The ridge precision is 5 for attack/defence and 1 for intercept/home advantage. Parameter bounds stabilize fitting: intercept [−4, 4], home advantage [−1, 1], attack/defence [−2, 2]. These are fixed starting choices, not hyperparameters selected on the evaluation data. Unsuccessful optimization emits a warning and retains the previous fitted model.

Cold-start away means are 1.2 goals, 10 shots and 3.5 SOT; the initial home multiplier is 1.25. Unseen teams have neutral attack/defence effects. Published `*_attack_rating` / `*_defence_rating` values are exponentiated effects: 1 is neutral and a defence value above 1 means more conceded. These models adjust for the opponent through its corresponding attack or defence parameter. The shot models estimate count means; they are not an exact reproduction of GAP ratings.

`expected_home_goals` and `expected_away_goals` are the goal model's lambdas; `expected_total_goals` and `expected_goal_diff` are their sum and difference. These are predictions of match goals, not shot-location xG. H/D/A probabilities use the Skellam distribution of the difference of two independent Poisson variables, without a truncated score grid. This implementation **does not** include the Dixon–Coles low-score correlation correction.

## Evaluation

The CLI and dashboard's **Run walk-forward comparison** button compare feature families using the same labeled rows and date folds. The last 15% of distinct dates are excluded from this comparison. The first half of the remaining dates forms the initial training window; three successive evaluation blocks use expanding training histories. Earlier completed evaluation matches may update team features and count models, matching daily predictions, while the classifier and its preprocessor are refitted only at fold boundaries. This does not represent predicting an entire season in advance.

Outputs are fold-level log loss, accuracy, multiclass Brier score (sum across three classes, range 0–2), and ten-bin calibration tables per H/D/A outcome. Aggregate metrics are weighted by evaluated match count. Calibration tables diagnose probabilities; they do not fit a probability calibrator. Use development folds to select groups and parameters, then evaluate the choice on an untouched later period. Differences between groups are descriptive; no significance claim is made.

The dashboard's separate model comparison uses calendar train/validation/test windows (through 2022, 2023–24, 2025 onward), or a chronological 65/20/15 split by distinct dates if those windows are unavailable. Dates never cross split boundaries, and insufficient dates raise an error instead of reusing matches in multiple sets. Bookmaker edge analysis uses test rows and real valid odds only; without odds, it is skipped. The current public CSV has no bookmaker odds columns.

## Individual features and the dummy baseline

The dashboard's **Features: förklaring och värde mot dummy** view lists all 168 canonical inputs (including League). Click a row or select a feature to read its definition, formula, availability assumptions, aliases and model groups. `src/feature_catalog.py` documents all 182 exported columns including compatibility aliases. `home_position` and `away_position` already represent the league-season table **before each matchday**; they are not live standings. The loaded date range is shown so a small historical CSV sample cannot be mistaken for current fixtures.

`src/feature_diagnostics.py` fits a separate logistic regression for each individual feature, with `C=1` and at most 3,000 iterations. Numeric imputation/scaling and categorical one-hot encoding are fitted on training rows only. No other features enter these models. Every feature uses the same chronological train/test split as the dashboard models, and validation rows are excluded. Constant or entirely missing training features use the dummy predictions and are explicitly labeled.

The dummy always predicts the **training-set frequencies** of H, D and A for every test match, equivalent to `DummyClassifier(strategy="prior")`. It does not estimate priors from test results. The three dummy probabilities and the observed test frequencies are displayed separately. Dummy accuracy is the test accuracy of always choosing the most frequent training outcome.

For each outcome, Gini is `2 × ROC AUC − 1` against the two other outcomes. Macro Gini is the unweighted mean of the three Ginis. A constant dummy has Gini 0 whenever the outcome and its complement both occur in test. If an outcome is absent, its Gini and the three-outcome macro Gini are undefined, not zero. This is AUC-based predictive Gini, not random-forest node impurity. See the [scikit-learn ROC AUC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).

`log_loss_gain = dummy log loss − feature log loss`: positive means better probability predictions than dummy on those test rows. The view reports overall multiclass log loss and separate binary log losses for H-vs-rest, D-vs-rest and A-vs-rest, plus accuracy and missing-data rates. Downloadable CSV results also include Brier scores and improvements. High Gini does not itself imply good calibration or betting profits, and standalone performance does not measure a feature's incremental value alongside other inputs. Using this test view to select features makes the viewed period development data; subsequent final evaluation needs a new untouched period.

Remote CSV loading now uses Requests with its CA trust store and normal HTTPS verification, rather than pandas' urllib download path, to avoid the local macOS certificate-store error. Downloads have connection/read timeouts and HTTP errors are checked before parsing. The dashboard shows an actionable error if downloading fails. TLS verification is never disabled. See [Requests certificate verification](https://requests.readthedocs.io/en/latest/user/advanced/#ssl-cert-verification).

## Model lab and feature-combination simulations

The sidebar now switches between **Modellering** (the default) and **Features**. The model page begins with a logistic model using all canonical variables alongside the training-prior dummy. It shows H/D/A mean predicted probabilities, observed frequencies, accuracy, per-outcome and macro Gini, log loss and Brier on both train and test. The same fitted model evaluates both periods. Mean predicted probabilities can resemble the dummy's class frequencies even when individual match predictions improve; these percentages are not accuracy.

Feature screening uses the first 80% of distinct **training** dates to fit each individual logistic model and the last 20% of training dates to score it. The external validation and test periods do not determine this filter. By default, a feature must have macro Gini **strictly greater than 0.02** and a converged fit to pass. The threshold is adjustable in the sidebar. Constant/missing features, undefined Gini and failed convergence do not pass. Both pages mark rejected feature names red; the feature page can hide them. This is a standalone screening heuristic and can exclude features that would help through joint effects.

On the model page, the filter is enabled by default. A second logistic model uses all surviving features. Turning the filter off makes all canonical inputs available for the combination search. The search defaults to **50 unique combinations**, **5–20 features** per model, and seed **42**, with controls for 1–1,000 simulations, feature-count bounds and seed. Bounds must fit the available feature pool; the UI explains invalid settings. If fewer unique combinations exist, it runs all available combinations and reports the actual count.

Combinations are sampled without feedback from evaluation scores. For large search spaces, the feature count is sampled uniformly between the chosen bounds, followed by a uniformly sampled subset of that size; duplicates are rejected. Small spaces (at most 10,000 possible subsets) are enumerated and sampled without replacement. The fixed seed reproduces the combinations. Every classifier uses the same train, validation and test dates, with `C=1` and a 3,000-iteration limit. Fits that do not converge are reported and excluded from winner selection.

Results can be ranked by **test or validation**, using macro Gini (higher), log loss (lower), or Brier (lower). The default is test Gini as requested. Choosing the best among many test scores makes that test period model-selection data, so the displayed winner is exploratory rather than an unbiased final estimate. A new untouched period is needed for a final performance claim. Previous search results persist across normal reruns, and changing settings labels them as belonging to the previous run; changed input data invalidates their display.

Plots show train/validation/test Gini for every simulation and the best test Gini observed so far. Each point is a separately trained model, not a training iteration. Select any simulated model to inspect its feature list and dummy comparison, per-outcome ROC curves, and calibration curves. Gini is computed using all observations; only plotted ROC lines may be downsampled. Calibration tables include bin counts. A confidence threshold compares mean maximum predicted probability with actual accuracy among the selected matches on train and test, and individual test probabilities are downloadable. These diagnostics do not recalibrate probabilities or establish betting profitability.

Core experiment logic is in `src/model_experiments.py`; the interactive view is in `src/model_lab.py`. The verification suite includes train/test isolation, identical evaluation rows, reproducible unique combinations, feature-count limits, a 50-model search, and exclusion of invalid winners.

## Method references

Elo-based football covariates were evaluated by [Hvattum and Arntzen (2010)](https://www.sciencedirect.com/science/article/pii/S0169207009001708). Poisson goal modelling and time weighting have established football applications, including [Dixon and Coles (1997)](https://doi.org/10.1111/1467-9876.00065). Statistics-based ratings are studied by [Wheatcroft, Forecasting football matches by predicting match statistics](https://arxiv.org/abs/2001.09097). These motivate the feature families; they do not validate this implementation's precise priors, half-lives, regularization or feature combinations.
