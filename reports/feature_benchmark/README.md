# Feature benchmark — 2026-09-12

Run on all 45,907 rows of the project's public CSV, covering 2000-07-28 through 2026-01-19 across five leagues. This is a comparison on development folds, not a final holdout score or a profitability backtest.

```bash
python -m src.evaluate_features --data /tmp/betting-optimizer-matches.csv --output reports/feature_benchmark
```

Dataset SHA-256: `65ef51c5fe81cf9b1cd49ef25fdb04de957d9069ce58b149b67887237dabe0e2`.

Environment: Python 3.11, pandas 3.0.5, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.1. Hyperparameters were fixed before this run; no feature selection or tuning used the final holdout.

All feature families use the same 18,659 evaluation matches across three expanding folds. The final period beginning **2022-05-17** is excluded from these benchmark scores. Ratings use completed matches from earlier dates, including earlier evaluation dates, as they would for daily prediction.

| Feature group | Log loss ↓ | Brier score ↓ | Accuracy ↑ |
| --- | ---: | ---: | ---: |
| `baseline_context` | 0.981182 | 0.583082 | 53.01% |
| `baseline_shot_ratings` | 0.981492 | 0.583762 | 52.78% |
| `corrected_legacy` | 0.982199 | 0.583918 | 52.85% |
| `all_features` | 0.982909 | 0.583826 | 52.77% |
| `baseline_poisson` | 0.983864 | 0.585569 | 52.61% |
| `extended_stats` | 0.984382 | 0.585904 | 52.53% |
| `baseline_form` | 0.986371 | 0.587349 | 52.41% |

`baseline_context` and `baseline_shot_ratings` improve slightly over the corrected baseline on these development folds. Using every feature does not give the best result. These small aggregate differences do not establish statistical significance or performance on newer seasons. Calibration diagnostics are in `calibration.csv`; fold-by-fold scores and exact dates are in `fold_metrics.csv`. Evaluate a development-selected configuration on an untouched later period before drawing stronger conclusions.

The independent-Poisson goal model and the shot count models are documented in the main README. This comparison does not isolate individual context features and does not compare against the previous buggy implementation.
