# VolatilityForge — Decision Log

## D001 — Project Scope

**Decision:**

Forecast market volatility rather than price direction.

**Reason:**

The project evaluates volatility forecasting methods and their potential use in volatility-targeted portfolio strategies.

**Status:** Accepted

## D002 — Initial Asset

**Decision:**

Use SPY as the initial core asset.

**Reason:**

Establish a reproducible baseline before expanding to other assets.

**Status:** Accepted

## D003 — Evaluation Period Classification

**Decision:**

Treat 2022 onward as the development and historical evaluation period, not as an untouched final test set.

**Reason:**

Model comparisons, model corrections, regime analysis, and strategy backtests have already used this period. Reusing it as a final holdout would overstate the independence of the evaluation.

**Future protocol:**

Define and document a prospective evaluation period after model specifications, feature definitions, hyperparameters, and evaluation procedures are frozen. Do not use results from that period to select or tune models.

**Status:** Accepted

## D004 — Evaluation Protocol

**Decision:**

Use expanding-window walk-forward evaluation.

**Reason:**

Preserve chronological order and simulate forecasting with information available at each prediction date.

Training labels and residuals must not be used before their outcomes are available. For five-day targets, the outcome must be fully matured before the corresponding label or residual becomes eligible.

**Status:** Accepted

## D005 — Primary Benchmark

**Decision:**

Use HAR-RV as a key benchmark for volatility forecasts.

**Reason:**

Machine-learning models must be compared against a meaningful volatility-specific baseline.

**Status:** Accepted

## D006 — Initial Runtime

**Decision:**

Start with the available Python 3.14.7 environment and verify dependency compatibility before relying on it.

**Reason:**

Scientific and quantitative-finance packages may have version-specific compatibility constraints.

**Status:** Pending dependency verification

## D007 — Five-Day HAR-RV OOS Smearing Experiment

**Decision:**

Keep the out-of-sample smearing correction experimental. Do not adopt it in the production HAR-RV model yet.

**Method:**

Estimate Duan's smearing factor from historical out-of-sample log residuals. A residual originating at position `s` is eligible at forecast origin `t` only when:

`s + 5 < t`

This ensures the residual's five-day target has fully matured before it is used for the correction.

**Development results:**

| Metric | Baseline HAR-RV | OOS-smearing HAR-RV |
|---|---:|---:|
| Mean QLIKE | 0.4576349112 | 0.3727666845 |
| MSE | 1.0178967877 × 10⁻⁶ | 9.9173462053 × 10⁻⁷ |
| Observations | 1,196 | 1,196 |

- Evaluation period: January 3, 2022, through October 8, 2026.
- Mean out-of-sample smearing factor: approximately 1.450930.
- Aggregate QLIKE improvement: approximately 18.54%.
- Aggregate MSE improvement: approximately 2.57%.
- QLIKE improved on 496 of 1,196 dates.

**Findings:**

The correction improves aggregate QLIKE and MSE in the development sample, but its performance is uneven across years and realized-variance regimes. Low-realized-variance regime performance deteriorates. The smearing factor is relatively stable, but the effect of applying it varies with the realized outcome and baseline forecast.

**Limitations:**

These are development-period results, not an untouched holdout evaluation. Five-day targets overlap, so daily forecast losses are dependent. The descriptive results do not establish statistical significance or future performance.

**Next action:**

Freeze model specifications, feature definitions, hyperparameters, and the evaluation protocol before defining a prospective evaluation period. Do not use prospective results for model selection or tuning.

**Status:** Experimental; production model unchanged.