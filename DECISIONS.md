# VolatilityForge — Decision Log

## D001 — Project Scope

Decision:
Forecast market volatility rather than price direction.

Reason:
The project evaluates volatility forecasting methods and their
potential use in volatility-targeted portfolio strategies.

Status: Accepted

## D002 — Initial Asset

Decision:
Use SPY as the initial core asset.

Reason:
Establish a reproducible baseline before expanding to other assets.

Status: Accepted

## D003 — Evaluation Period Classification

Decision:
Treat 2022 onward as the development and historical evaluation period, not as an untouched final test set.

Reason:
Model comparisons, model corrections, regime analysis, and strategy backtests have already used this period. Reusing it as a final holdout would overstate the independence of the evaluation.

Future protocol:

Define and document a prospective evaluation period after model specifications, feature definitions, hyperparameters, and evaluation procedures are frozen. Do not use results from that period to select or tune models.

Status: Accepted

## D004 — Evaluation Protocol

Decision:
Use expanding-window walk-forward evaluation.

Reason:
Preserve chronological order and simulate forecasting with
information available at each prediction date.

Status: Accepted

## D005 — Primary Benchmark

Decision:
Use HAR-RV as a key benchmark for volatility forecasts.

Reason:
Machine-learning models must be compared against a meaningful
volatility-specific baseline.

Status: Accepted

## D006 — Initial Runtime

Decision:
Start with the available Python 3.14.7 environment and verify
dependency compatibility before relying on it.

Reason:
Scientific and quantitative-finance packages may have
version-specific compatibility constraints.

Status: Pending dependency verification
