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
- Final holdout period: 2022 onward
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

Phase 1: Repository setup.

No forecasting performance claims have been established yet.

## Reproducibility

Experiments will use explicit date ranges, documented modeling
decisions, automated tests, and time-ordered evaluation.

The final test period will not be used for model selection or tuning.

## Disclaimer

This project is for educational and research purposes only.
Backtest results do not guarantee future performance and are not
financial advice.
