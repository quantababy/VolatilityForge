# VolatilityForge Evaluation Protocol

**Status:** Draft — pending verification and approval  
**Protocol version:** 1.0  
**Primary asset:** SPY  
**Implied-volatility reference:** VIX

## 1. Research objectives

Forecast next-day and next-five-trading-day realized variance. Evaluate forecasts using chronological out-of-sample predictions and compare them with established statistical benchmarks.

The primary objective is forecast quality, not directional price prediction.

## 2. Forecast targets

- **One-day horizon:** squared return over the next trading session.
- **Five-day horizon:** sum of squared returns over the next five trading sessions.

Forecast origins, target dates, and realized outcomes must remain explicitly aligned.

## 3. Information availability and leakage prevention

Use expanding-window walk-forward evaluation.

At each forecast origin, training data may contain only labels whose outcomes have fully matured before that origin.

For a five-day target, a forecast originating at position `s` is eligible for training or residual estimation at position `t` only when `s + 5 < t`, under the project's current indexing convention.

All preprocessing, feature selection, model fitting, and residual-based corrections must respect the same chronological information boundary.

## 4. Evaluation periods

- **Historical forecast initialization:** configured in `configs/config.yaml`.
- **Development evaluation:** January 2022 onward, as specified by the configuration.
- **Prospective final evaluation:** not yet designated.

The development period has already been used for model comparison, statistical analysis, regime analysis, and strategy experiments. It must not be described as an untouched holdout.

A prospective evaluation period may be designated only after model specifications, features, hyperparameters, metrics, and evaluation procedures are frozen. Prospective results must not be used for model selection or tuning.

## 5. Metrics

- **Primary metric:** normalized QLIKE on strictly positive realized-variance targets.
- **Secondary metric:** mean squared error (MSE), evaluated on valid nonnegative targets.

Report the sample size and date range for each metric. If metric-specific valid samples differ, report those differences explicitly.

## 6. Model comparisons and statistical inference

Use paired Diebold–Mariano comparisons of forecast losses.

For five-day forecasts, HAC lag 4 is the primary configured comparison; lags 5, 10, and 22 are sensitivity analyses.

For one-day forecasts, report the configured HAC-lag sensitivity analyses.

Apply Holm correction across competitor comparisons within each lag family. Report raw and adjusted p-values, sample sizes, and the inference method. Interpret lag sensitivity transparently rather than selecting whichever lag produces significance.

The five-day forecast losses overlap, so inference must account for serial dependence. Statistical corrections do not guarantee validity under arbitrary dependence or model misspecification.

## 7. Benchmarks and model selection

Maintain statistical baselines before selecting more complex models. HAR-RV is a key volatility-specific benchmark.

Any model selection or correction proposed from development-period results must be identified as development-stage research.

## 8. Experimental corrections

The five-day HAR-RV smearing correction remains experimental. It must not silently replace the production HAR-RV forecast.

Any future adoption requires a documented decision, leakage-safe residual construction, and a fresh evaluation under a previously frozen protocol.

## 9. Strategy evaluation

Report transaction costs and portfolio-risk controls alongside strategy performance. Historical strategy results are exploratory and do not establish future profitability.

## 10. Reproducibility and change control

Record configuration, random seeds where applicable, data dates, model specifications, metric definitions, and software dependencies for each experiment.

Any change to forecast targets, features, hyperparameters, model selection rules, metrics, or statistical procedures must be documented before conducting a designated prospective evaluation.

## 11. Current limitations

- The 2022-onward period is development data, not an untouched final holdout.
- The prospective final evaluation has not been designated.
- The smearing experiment is exploratory.
- Python and scientific-package compatibility verification remains an explicit project decision.
- Historical performance does not establish future forecast quality or strategy profitability.
