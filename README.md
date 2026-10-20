# VolatilityForge

A quantitative finance project for forecasting market volatility using statistical models, econometrics, and machine learning.

## Objectives

- Forecast next-day realized variance.
- Forecast cumulative realized variance over the next five trading days.
- Establish statistical baselines before introducing machine learning.
- Evaluate models using leakage-free, expanding-window walk-forward validation.
- Compare forecasts using QLIKE as the primary metric and MSE as a secondary metric.
- Test volatility-targeting strategies with transaction costs.
- Provide reproducible experiments, documented decisions, and an interactive dashboard.

## Initial Scope

- **Core asset:** SPY
- **Implied volatility reference:** VIX
- **Development evaluation period:** 2022 onward
- **Prospective final evaluation:** To be defined after the evaluation protocol is frozen
- **Primary benchmark:** HAR-RV

## Planned Models

- EWMA
- GARCH and GJR-GARCH
- HAR-RV
- Ridge and Lasso
- LightGBM and XGBoost
- HAR-RV residual hybrid

## Repository Structure

- `configs/`: Configuration and evaluation settings
- `data/`: Raw and processed market data
- `src/`: Data processing, forecasting models, evaluation, and application code
- `notebooks/`: Exploratory analysis
- `tests/`: Automated tests
- `reports/`: Forecast comparisons, statistical tests, and experiment results
- `scripts/`: Reproducible analysis and experiment utilities
- `dashboard/`: Planned interactive dashboard


## Rebuilding the Data Pipeline

From the repository root, activate the project virtual environment and run:

```powershell
python -m src.run_data_pipeline
```

The pipeline reads `configs/config.yaml`, downloads the configured SPY and
VIX daily data, validates and saves the raw datasets, aligns both assets on
shared dates, and rebuilds the volatility features and forward targets.

Generated files:

- `data/raw/spy.parquet`
- `data/raw/vix.parquet`
- `data/processed/spy_vix.parquet`
- `data/processed/volatility_features.parquet`

The command prints row counts and date ranges for each output. It requires
network access for the market-data download. Its integration test mocks
downloads and runs without network access.

The pipeline rebuilds the raw Parquet files from the data provider's current
response; it does not guarantee bit-for-bit historical reproducibility if
the provider revises its data.


## Project Status

### Implemented

- Historical market-data loading using SPY and VIX data
- SPY/VIX date alignment
- Leakage-aware volatility feature and target construction
- EWMA, GARCH, GJR-GARCH, HAR-RV, and LightGBM forecasting implementations
- One-day and five-day forecast comparisons
- QLIKE and MSE evaluation
- Diebold–Mariano forecast comparisons and HAC-lag sensitivity analysis
- Volatility-regime analysis
- Volatility-targeting strategy backtests with transaction costs
- Automated tests for key forecasting and evaluation components

### Experimental Research

A separate five-day HAR-RV experiment evaluates Duan's smearing correction using historical out-of-sample log residuals.

The correction estimates a multiplicative factor from residuals whose five-day targets have fully matured before the forecast origin. The experiment is implemented separately from the production HAR-RV forecasting code.

Development-period results:

| Metric | Baseline HAR-RV | OOS-smearing HAR-RV |
|---|---:|---:|
| Mean QLIKE | 0.457635 | 0.372767 |
| MSE | 1.017897 × 10⁻⁶ | 9.917346 × 10⁻⁷ |
| Observations | 1,196 | 1,196 |

Evaluation period: January 3, 2022, through October 8, 2026.

The correction improved aggregate QLIKE by approximately 18.54% and MSE by approximately 2.57% in this development sample.

However, the improvement was uneven across years and realized-variance regimes. Low-realized-variance regime performance deteriorated, and QLIKE improved on 496 of 1,196 individual dates.

These results are exploratory. The smearing correction has **not** been adopted in the production HAR-RV model. The results do not establish statistical significance or future performance.

### Next Priorities

1. Freeze model specifications, feature definitions, hyperparameters, and the evaluation protocol.
2. Define a prospective evaluation period after the protocol is frozen.
3. Preserve the prospective period for final evaluation rather than model selection or tuning.
4. Improve experiment reproducibility, diagnostics, and reporting.
5. Complete remaining planned models, dashboard development, and documentation.

## Reproducibility and Evaluation Protocol

Experiments use chronological expanding-window walk-forward evaluation. Training labels must be available before each forecast origin, and multi-day targets must be fully matured before they are eligible for training or residual estimation.

The period beginning in 2022 has already been used for model selection, model comparisons, corrections, regime analysis, and strategy backtests. It is therefore classified as development data, not an untouched final holdout.

The latest processed market-data date is October 15, 2026. The latest valid five-day target in the current smearing experiment is October 8, 2026; the final five forecast origins do not yet have fully matured five-day outcomes.

A prospective evaluation period must be defined after model specifications, feature definitions, hyperparameters, and evaluation procedures are frozen. Results from that period must not be used for model selection or tuning.

## Disclaimer

This project is for educational and research purposes only. Historical backtest and forecast-comparison results do not guarantee future performance and are not financial advice.