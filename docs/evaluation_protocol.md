# VolatilityForge Evaluation Protocol

**Status:** Draft — pending verification and approval  
**Protocol version:** 1.0  
**Primary asset:** SPY  
**Implied-volatility reference:** VIX

## 1. Research Objectives

Forecast next-day and next-five-trading-day realized variance using chronological out-of-sample predictions. Evaluate forecast quality against established statistical benchmarks and assess whether more complex models provide demonstrable improvements.

The primary objective is volatility forecasting, not directional price prediction.

All reported results must be reproducible and must distinguish historical development findings from prospective evaluation results.

## 2. Forecast Targets and Alignment

- **One-day horizon:** Squared return over the next trading session.
- **Five-day horizon:** Sum of squared returns over the next five trading sessions.

Forecast origins, target dates, realized outcomes, and model predictions must remain explicitly aligned. The forecast horizon must not be confused with the date on which its target becomes observable.

Forecasts must be strictly positive wherever required by the evaluation metrics. Invalid or unavailable forecasts must be excluded according to a consistent, documented sample-selection rule.

## 3. Information Availability and Leakage Prevention

Use expanding-window walk-forward evaluation.

At each forecast origin, training data may contain only labels whose outcomes are observable by that origin.

For one-day models, the target at position `s` represents the return over the next session. At the close of position `t`, the previous session's target may be used once the return at `t` has been observed. Therefore, one-day training labels must not use information that remains unavailable at the current close.

For five-day models, retain the project's conservative eligibility rule: a label originating at position `s` may be used for training or residual estimation at position `t` only when:

`s + 5 < t`

This rule is intentionally stricter than merely requiring the five-day outcome to have reached its final date.

The same chronological information boundary must apply to:

- Feature construction and preprocessing.
- Model fitting and hyperparameter selection.
- Residual estimation and bias corrections.
- Forecast comparisons and statistical inference.
- Any strategy signal generated from forecasts.

Any transformation or correction that estimates parameters from historical data must use only information eligible at the relevant forecast origin. Future observations must not influence historical predictions.

## 4. Evaluation Periods and Data Governance

- **Historical forecast initialization:** As specified by `configs/config.yaml`.
- **Development evaluation:** January 2022 onward, as specified by the project configuration and evaluation procedures.
- **Prospective final evaluation:** Not yet designated.

The development period has already been used for model comparisons, statistical tests, regime analysis, and strategy experiments. It is therefore **development data, not an untouched final holdout**.

A prospective evaluation period may be designated only after the protocol, model specifications, features, hyperparameters, metric definitions, and statistical procedures have been frozen and documented.

Once designated, prospective results must not be used for model selection, hyperparameter tuning, or repeated experimentation while continuing to describe the period as an untouched evaluation.

Any change to the frozen protocol must be documented. If the change is informed by prospective results, the affected evaluation must be identified accordingly; it cannot continue to be presented as an untouched confirmatory test.

## 5. Metrics and Sample Selection

### Primary metric: normalized QLIKE

Use normalized QLIKE on strictly positive realized-variance targets and strictly positive forecasts:

`QLIKE = target / forecast - log(target / forecast) - 1`

Lower values indicate better performance.

Zero realized-variance targets are excluded from QLIKE because the ratio and logarithm require a strictly positive target. This exclusion must be disclosed when applicable.

### Secondary metric: mean squared error

Use mean squared error (MSE) on valid nonnegative realized-variance targets and valid forecasts:

`MSE = mean((target - forecast)^2)`

Zero targets may be included in MSE.

### Reporting requirements

For every metric, report:

- Forecast horizon and model.
- Evaluation date range.
- Number of observations.
- Metric value and relevant comparison baseline.
- Any metric-specific exclusions or differences in the valid sample.

When comparing models, use a common valid sample for the comparison being reported. If QLIKE and MSE use different samples, report their sample sizes and date ranges separately.

Do not compare metric values across different samples as though they were directly comparable.

## 6. Model Comparisons and Statistical Inference

Use paired Diebold–Mariano (DM) tests to compare forecast-loss series on aligned observations.

The comparison scripts pre-specify the following HAC lag settings:

- **Five-day forecasts:** HAC lag 4 is the primary comparison setting; lags 5, 10, and 22 are sensitivity analyses.
- **One-day forecasts:** HAC lags 5, 10, and 22 are sensitivity analyses.

These settings are pre-specified in the comparison scripts; they should not be described as YAML-configured settings unless the implementation is changed to load them from the configuration file.

Apply Holm correction across competitor comparisons within each HAC-lag family. Report raw and adjusted p-values, sample sizes, the test method, and the HAC lag used.

The five-day forecast losses overlap, so inference must account for serial dependence. Lag sensitivity must be reported transparently; do not select a lag solely because it produces statistical significance.

Statistical significance does not establish economic significance, prove that a model is universally superior, or guarantee validity under arbitrary dependence or model misspecification.

Any change to the pre-specified lag settings, comparison family, or correction procedure must be documented before a designated prospective evaluation.

## 7. Benchmarks and Model Selection

Maintain simple and established statistical baselines before selecting more complex models.

The project's benchmark set includes:

- EWMA volatility forecasting.
- GARCH.
- GJR-GARCH.
- HAR-RV.
- LightGBM as a machine-learning candidate.

HAR-RV is a key volatility-specific benchmark. EWMA's five-day forecast is an approximation based on five times its one-day variance forecast; it assumes constant expected daily variance over the forecast horizon and must be described accordingly.

Evaluate additional model complexity against the benchmarks using the same eligible forecast origins, target definitions, and valid-sample rules wherever applicable.

Model selection and conclusions based on the existing development period must be identified as development-stage findings. The best observed development-period result must not be presented as proof of future superiority.

## 8. Experimental Corrections

The five-day HAR-RV smearing correction remains experimental and must not silently replace the baseline HAR-RV forecast.

Any proposed adoption requires:

1. A documented change decision.
2. Leakage-safe residual construction using only eligible, matured outcomes.
3. A clearly specified estimation procedure.
4. A comparison against the unchanged baseline on aligned observations.
5. Appropriate uncertainty analysis and a fresh evaluation under a previously frozen protocol.

Development-period improvements are exploratory. They do not, by themselves, establish statistical significance, generalization to future observations, or suitability for production.

## 9. Strategy Evaluation

Historical strategy results are exploratory and do not establish future profitability.

Strategy evaluations must document:

- Forecast-to-position mapping and execution timing.
- Position sizing and annualized volatility target.
- Leverage limits and other risk controls.
- Transaction-cost assumptions and their application.
- Portfolio returns, volatility, drawdown, turnover, and other relevant risk measures.
- Performance before and after transaction costs.
- The benchmark and evaluation period.

The current configuration specifies a 10% annualized volatility target, maximum leverage of 1.0, and transaction costs of 5 basis points. Report these as assumptions, not as evidence that the strategy is executable or profitable in live markets.

No strategy result should be described as robust without appropriate sensitivity analysis. Historical backtests do not guarantee future performance.

## 10. Reproducibility and Change Control

Record the following for each reported experiment:

- Configuration and random seed, where applicable.
- Data source, sample dates, and data version or retrieval details.
- Forecast target and horizon definitions.
- Feature set and model specifications.
- Training and forecast-origin eligibility rules.
- Metric definitions and valid-sample rules.
- Statistical-test settings and multiple-testing correction.
- Software versions and dependency information.
- Experiment outputs and the code revision used.

Changes to targets, features, hyperparameters, model-selection rules, metrics, comparison procedures, or statistical settings must be documented.

The protocol and implementation must agree. If a documented setting differs from the code, resolve the discrepancy before treating the corresponding experiment as protocol-compliant.

## 11. Current Limitations and Approval Conditions

- The period beginning in January 2022 is development data, not an untouched final holdout.
- A prospective final evaluation period has not been designated.
- The five-day HAR-RV smearing correction remains exploratory.
- The five-day EWMA forecast is an approximation based on constant expected daily variance.
- HAC lag settings are pre-specified in comparison scripts rather than loaded from YAML configuration.
- Python and scientific-package compatibility verification remains an explicit project decision.
- Historical forecast accuracy and strategy returns do not establish future performance.
- The protocol remains a draft until the wording, implementation, dependencies, and evaluation procedures have been verified and approved.

**Approval requirement:** Before designating a prospective final evaluation, verify this protocol against the implementation, freeze the relevant specifications, document the code revision and dependencies, and record the approval decision.
