# VolatilityForge

A quantitative finance project for forecasting market volatility using
statistical models, econometrics, and machine learning.

## Objectives

- Forecast next-day realized volatility.
- Forecast the mean realized volatility over the next five trading days.
- Establish statistical baselines before introducing machine learning.
- Evaluate models using leakage-free walk-forward validation.
- Compare forecasts using QLIKE and MSE.
- Test volatility-targeting strategies with transaction costs.
- Provide reproducible experiments and an interactive dashboard.

## Initial Scope

- Core asset: SPY
- Implied volatility reference: VIX
- Development evaluation period: 2022 onward
- Prospective final evaluation: to be defined after the evaluation protocol is frozen
- Primary benchmark: HAR-RV

## Planned Models

- EWMA
- GARCH and GJR-GARCH
- HAR-RV
- Ridge and Lasso
- LightGBM and XGBoost
- HAR-RV residual hybrid

## Repository Structure

- `configs/`: configuration files
- `data/`: raw and processed market data
- `src/`: application and modeling code
- `notebooks/`: exploratory analysis
- `tests/`: automated tests
- `reports/`: project reports
- `dashboard/`: interactive dashboard

## Project Status

Implemented:
- Historical market-data loading and SPY/VIX alignment
- Leakage-aware volatility feature and target construction
- EWMA, GARCH, GJR-GARCH, HAR-RV, and LightGBM forecasts
- One-day and five-day forecast comparisons
- QLIKE/MSE evaluation, Diebold-Mariano comparisons, and regime analysis
- Volatility-targeting strategy backtests with transaction costs
- Automated tests for key forecasting and evaluation components

Current evaluation results use the development period beginning in 2022.
They are historical research results, not evidence of performance on an
untouched final test set.

Next priorities:
- Freeze model specifications and the evaluation protocol
- Define a prospective evaluation period
- Improve experiment reproducibility and reporting
- Complete remaining planned models, dashboard, and documentation

## Reproducibility

Experiments use chronological walk-forward evaluation, documented modeling
decisions, explicit evaluation settings, and automated tests.

The period beginning in 2022 has already been used for model selection,
comparisons, corrections, regime analysis, and strategy backtests. It is
therefore classified as development data, not an untouched final holdout.

A prospective evaluation period must be defined after model specifications,
feature definitions, hyperparameters, and evaluation procedures are frozen.
Its results must not be used for model selection or tuning.

## Disclaimer

This project is for educational and research purposes only.
Backtest results do not guarantee future performance and are not
financial advice.
