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

## D003 — Final Test Period

Decision:
Reserve 2022 onward as the final out-of-sample test period.

Reason:
Prevent model selection and hyperparameter tuning from using
the final evaluation period.

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
