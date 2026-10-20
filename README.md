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

The pipeline reads `configs/config.yaml`, downloads the configured SPY and VIX daily data, validates and saves the raw datasets, aligns both assets on shared dates, and rebuilds the volatility features and forward targets.

Generated files:

- `data/raw/spy.parquet`
- `data/raw/vix.parquet`
- `data/processed/spy_vix.parquet`
- `data/processed/volatility_features.parquet`

The command prints row counts and date ranges for each output. It requires network access for the market-data download. Its integration test mocks downloads and runs without network access.

The pipeline rebuilds the raw Parquet files from the data provider's current response; it does not guarantee bit-for-bit historical reproducibility if the provider revises its data.

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
- Strategy performance reports with gross/net returns and a same-period gross buy-and-hold benchmark
- Strategy sensitivity analysis across 8%, 10%, and 12% annual volatility targets and 0, 5, 10, and 20 bps transaction costs
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

### Strategy Economic Evaluation

The strategy backtests apply each forecast made after close on date *t* to the next session's return. Exposure is long-only and capped by the configured maximum leverage. Transaction costs are charged on absolute changes in exposure.

Reports include gross and net returns, annualized return and volatility, zero-risk-free-rate Sharpe ratio, maximum drawdown, exposure, turnover, and a same-period 1x gross buy-and-hold benchmark.

Generated reports:

- `reports/strategy_performance.csv`
- `reports/strategy_sensitivity.csv`

The sensitivity report evaluates five forecasting models at annualized volatility targets of 8%, 10%, and 12%, with transaction costs of 0, 5, 10, and 20 basis points. It reports net performance and turnover-related costs. The configured baseline is a 10% volatility target, 1.0 maximum leverage, and 5 basis points per unit of turnover.

#### Reproducing the Strategy Reports

Run the following commands from the repository root with the virtual environment activated:

```powershell
python -m src.run_strategy_backtest
python -m src.run_strategy_report
python -m src.run_strategy_sensitivity
```

The commands perform the following tasks:

1. `src.run_strategy_backtest` runs the configured volatility-targeting backtests using the saved forecasts and processed feature data.
2. `src.run_strategy_report` generates `reports/strategy_performance.csv`.
3. `src.run_strategy_sensitivity` evaluates the five forecasting models across the three target-volatility levels and four transaction-cost assumptions, then writes `reports/strategy_sensitivity.csv`.

The processed feature data and forecast files must already exist. These commands do not independently rebuild the data pipeline or retrain forecasting models. The sensitivity runner reuses the project's strategy backtest and metrics implementations.

These results use the development period beginning in 2022. This period has already been used for model comparison and strategy analysis; it is not an untouched final holdout. Sensitivity analysis does not establish statistical robustness, live executability, or future profitability.

The buy-and-hold benchmark is gross, while strategy net returns deduct the configured costs, so those comparisons are not cost-matched. Market-data providers may revise historical observations, and a future prospective evaluation period must be defined only after the protocol and model specifications are frozen.

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
