# 📈 VolatilityForge: Quantitative Volatility Forecasting & Strategy Engine

Welcome to **VolatilityForge**! ⚡ This is an end-to-end quantitative finance and machine learning project designed to forecast equity market volatility using statistical baselines, econometric models, and gradient boosting— paired with leakage-free walk-forward evaluation and realistic volatility-targeting backtests.

---

## 🎯 Project Objectives & Scope

### 🧭 Core Objectives
* **📅 Multi-Horizon Forecasting:** Predict next-day ($1\text{-day}$) and cumulative five-trading-day ($5\text{-day}$) realized variance.
* **📊 Rigorous Baselines First:** Establish strong econometric benchmarks (`EWMA`, `GARCH`, `GJR-GARCH`, `HAR-RV`) before introducing machine learning.
* **🛡️ Zero-Leakage Validation:** Evaluate all models via chronological, expanding-window walk-forward validation where multi-day targets must fully mature prior to training.
* **⚖️ Loss-Function Precision:** Compare forecasts using **QLIKE** as the primary asymmetric metric and **MSE** as the secondary metric, backed by Diebold–Mariano tests and HAC-lag sensitivity analysis.
* **💼 Economic Backtesting:** Translate variance forecasts into actionable **volatility-targeting strategies** accounting for turnover and transaction costs.
* **🖥️ Interactive Research:** Deliver reproducible pipelines, documented research decisions, and an interactive **Streamlit** dashboard.

### 🔭 Initial Scope Parameters
| Parameter | Specification |
| :--- | :--- |
| **Core Asset** | `SPY` (S&P 500 ETF) |
| **Implied Volatility Reference** | `VIX` (CBOE Volatility Index) |
| **Primary Benchmark** | `HAR-RV` (Heterogeneous Autoregressive Realized Volatility) |
| **Development Evaluation Period** | `2022-01-03` onward *(classified as development data)* |
| **Prospective Final Evaluation** | To be defined *after* the evaluation protocol and model specs are frozen |

---

## 🤖 Forecasting Model Status

| Model Family | Specific Implementation | Status |
| :--- | :--- | :--- |
| **Exponential Smoothing** | `EWMA` | ✅ Implemented |
| **Econometric GARCH** | `GARCH(1,1)` & `GJR-GARCH` | ✅ Implemented |
| **Autoregressive Baseline** | `HAR-RV` (1d & 5d horizons) | ✅ Implemented |
| **Gradient Boosting** | `LightGBM` | ✅ Implemented |
| **Linear Regularization** | `Ridge` & `Lasso` | ⏳ *Pending Verification* |
| **Alternative Boosting** | `XGBoost` | ⏳ *Pending Verification* |
| **Hybrid Architecture** | `HAR-RV Residual Hybrid` | ⏳ *Pending Verification* |

---

## 📂 Repository Architecture

```text
📦 VolatilityForge
 ┣ 📂 configs/      # YAML configuration & walk-forward evaluation settings
 ┣ 📂 data/         # Raw (SPY, VIX) and processed Parquet feature stores
 ┣ 📂 src/          # Core data pipeline, forecasting models, backtester & metrics
 ┣ 📂 notebooks/    # Exploratory data analysis & prototyping
 ┣ 📂 tests/        # Automated unit & offline-mocked integration tests
 ┣ 📂 reports/      # CSV performance outputs, DM tests & sensitivity matrices
 ┣ 📂 scripts/      # Reproducible experiment utilities
 ┗ 📂 dashboard/    # Interactive Streamlit research application
```

---

## ⚙️ Rebuilding the Data Pipeline

From the repository root, activate your virtual environment and run:

```powershell
python -m src.run_data_pipeline
```

### 🔄 What This Pipeline Does:
1. Reads configuration settings from `configs/config.yaml`.
2. Downloads daily historical data for **SPY** and **VIX** *(requires active network connection; unit tests mock this step offline)*.
3. Validates schemas, aligns both assets on shared trading dates, and computes leakage-free realized volatility features and forward targets.
4. Prints row counts and exact date spans for every generated artifact.

### 💾 Generated Parquet Artifacts:
* `data/raw/spy.parquet` & `data/raw/vix.parquet`
* `data/processed/spy_vix.parquet`
* `data/processed/volatility_features.parquet`

> ⚠️ **Reproducibility Note:** Rebuilding the pipeline queries the data provider's live API response. Bit-for-bit historical reproducibility is not guaranteed if the upstream provider retroactively revises historical records.

---

## 🧪 Experimental Research: 5-Day HAR-RV Smearing Correction

A standalone experimental study evaluates **Duan's smearing correction** on $5\text{-day}$ `HAR-RV` log-scale forecasts using historical out-of-sample (OOS) log residuals. 

To prevent lookahead bias, the multiplicative smearing factor is estimated strictly from historical residuals whose $5\text{-day}$ targets have **fully matured** prior to the forecast origin.

### 📊 Development-Period Results (`2022-01-03` to `2026-10-08`)

| Metric | Baseline HAR-RV | OOS-Smearing HAR-RV | Relative Change |
| :--- | :---: | :---: | :---: |
| **Mean QLIKE** | `0.457635` | `0.372767` | **-18.54%** 🟢 |
| **MSE** | `1.017897 × 10⁻⁶` | `9.917346 × 10⁻⁷` | **-2.57%** 🟢 |
| **Valid Observations** | `1,196` | `1,196` | — |

### 🔍 Key Takeaways & Caveats
* **Regime Asymmetry:** While aggregate QLIKE improved by **~18.54%**, performance gains were concentrated in elevated volatility regimes. In low-realized-variance regimes, performance deteriorated, and QLIKE improved on only **496 of 1,196** individual dates (`41.5%`).
* **Production Status:** These findings remain exploratory. The smearing correction has **not** been merged into the production `HAR-RV` model, and results do not imply statistical significance.

---

## 💹 Strategy Economic Evaluation & Backtesting

Forecasts generated after market close on date $t$ determine target portfolio exposure for session $t+1$:
* **Constraints:** Long-only exposure capped at configured maximum leverage (`1.0x` baseline).
* **Frictions:** Transaction costs are deducted proportionally to absolute daily changes in portfolio weight (turnover).
* **Metrics Tracked:** Gross/Net returns, annualized return & volatility, Sharpe ratio ($R_f = 0$), max drawdown, exposure, turnover, and a $1\text{x}$ **gross** buy-and-hold SPY benchmark.

### 🛠️ Reproducing Strategy Reports

Ensure processed feature data and model forecasts already exist, then execute:

```powershell
# 1. Run volatility-targeting backtests using saved forecasts
python -m src.run_strategy_backtest

# 2. Generate core strategy performance table -> reports/strategy_performance.csv
python -m src.run_strategy_report

# 3. Run multi-parameter grid sensitivity -> reports/strategy_sensitivity.csv
python -m src.run_strategy_sensitivity
```

### 🎛️ Sensitivity Grid Parameters
The sensitivity runner tests **5 forecasting models** across:
* **Annualized Volatility Targets:** `8%`, `10%` *(baseline)*, and `12%`
* **Transaction Cost Assumptions:** `0 bps`, `5 bps` *(baseline)*, `10 bps`, and `20 bps` per unit of turnover

> 📌 **Evaluation Guardrails:** The `2022+` sample has been actively used for model comparisons and strategy analysis—it is **development data**, not an untouched holdout. Furthermore, the $1\text{x}$ buy-and-hold benchmark is reported **gross** of fees, whereas strategy net returns deduct transaction costs.

---

## 🖥️ Running the Interactive Dashboard

Launch the **Streamlit** research interface from the repository root:

```powershell
python -m streamlit run dashboard/app.py
```

**Included Views:**
1. 📈 **Executive Overview:** Dataset coverage and summary statistics.
2. 🏆 **Model Comparison:** Horizon-specific QLIKE/MSE tables and Diebold–Mariano matrices.
3. 🌊 **Volatility Regimes:** Conditional performance across calm and stressed market environments.
4. 💰 **Strategy Backtests:** Equity curves, drawdown profiles, and cost sensitivity.

---

## 🔒 Reproducibility, Timeline & Next Priorities

### 📅 Data Horizon Snapshot
* **Latest Processed Market Data:** `October 15, 2026`
* **Latest Matured 5-Day Target:** `October 8, 2026` *(the final 5 forecast origins do not yet have fully realized 5-day forward outcomes).*

### 🚀 Next Priorities
1. **Freeze Specifications:** Lock model architectures, feature sets, hyperparameters, and evaluation rules.
2. **Define Prospective Holdout:** Establish a dedicated post-freeze prospective evaluation window.
3. **Protect Holdout Integrity:** Restrict the prospective window strictly to final reporting (zero tuning or model selection).
4. **Enhance Diagnostics:** Expand automated test coverage, pipeline logging, and release documentation.

---

## ⚠️ Disclaimer

*This repository is strictly for **educational and quantitative research purposes**. Historical backtests, regime analyses, and out-of-sample forecast comparisons do not guarantee future market performance and do not constitute financial or investment advice.*
