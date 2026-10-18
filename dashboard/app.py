
"""VolatilityForge research dashboard."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"

st.set_page_config(
    page_title="VolatilityForge",
    page_icon="📈",
    layout="wide",
)

st.title("VolatilityForge")
st.caption(
    "SPY volatility research | Statistical models, machine learning, "
    "historical evaluation and strategy analysis"
)


@st.cache_data
def load_csv(filename: str) -> pd.DataFrame:
    path = REPORTS / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_parquet(filename: str) -> pd.DataFrame:
    path = PROCESSED / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def show_missing(message: str) -> None:
    st.info(message)


def show_metric_table(data: pd.DataFrame, title: str) -> None:
    st.subheader(title)
    if data.empty:
        show_missing("No report data is available.")
    else:
        st.dataframe(data, width='stretch', hide_index=True)


# Sidebar
st.sidebar.header("Dashboard")
page = st.sidebar.radio(
    "Section",
    [
        "Overview",
        "Model comparison",
        "Volatility regimes",
        "Strategy backtests",
    ],
)

one_day = load_parquet("prospective_forecasts_1d.parquet")
five_day = load_parquet("prospective_forecasts_5d.parquet")

# ---------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------
if page == "Overview":
    st.header("Latest prospective forecasts")

    if one_day.empty or five_day.empty:
        show_missing(
            "Prospective forecast files are missing. Run "
            "`python -m src.run_prospective_forecasts` first."
        )
    else:
        one_day["forecast_date"] = pd.to_datetime(one_day["forecast_date"])
        five_day["forecast_date"] = pd.to_datetime(five_day["forecast_date"])

        forecast_date = one_day["forecast_date"].max()
        five_date = five_day["forecast_date"].max()

        if forecast_date != five_date:
            st.warning(
                "The one-day and five-day forecasts have different origins. "
                "Review the output files before comparing them."
            )

        st.caption(
            f"Forecast origin: {forecast_date.date()} | "
            "Variance forecasts use decimal-return units."
        )

        cols = st.columns(4)
        latest_values = one_day.set_index("model")["variance_forecast"]

        for i, model in enumerate(
            ["EWMA", "GARCH", "GJR-GARCH", "HAR-RV"]
        ):
            if model in latest_values.index:
                cols[i].metric(model, f"{latest_values[model]:.8f}")

        st.subheader("Next-day variance forecasts")
        fig = px.bar(
            one_day.sort_values("variance_forecast"),
            x="model",
            y="variance_forecast",
            color="model",
            labels={
                "model": "Model",
                "variance_forecast": "Forecast variance",
            },
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')

        st.subheader("Five-session cumulative variance forecasts")
        fig = px.bar(
            five_day.sort_values("variance_forecast"),
            x="model",
            y="variance_forecast",
            color="model",
            labels={
                "model": "Model",
                "variance_forecast": "Cumulative variance forecast",
            },
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')

        st.caption(
            "The five-day EWMA estimate uses five times its one-day variance. "
            "LightGBM currently has no prospective five-day forecast."
        )

        with st.expander("View forecast tables"):
            show_metric_table(one_day, "Next-day forecast records")
            show_metric_table(five_day, "Five-day forecast records")

# ---------------------------------------------------------------------
# Historical model comparison
# ---------------------------------------------------------------------
elif page == "Model comparison":
    st.header("Historical model comparison")
    st.warning(
        "These are development-period results, not an untouched final "
        "holdout. Different report files may use different QLIKE conventions "
        "or samples; compare results within each report."
    )

    horizon = st.radio(
        "Forecast horizon",
        ["1 day", "5 days"],
        horizontal=True,
    )

    if horizon == "1 day":
        comparison = load_csv("volatility_model_comparison.csv")
        dm = load_csv("dm_test_sensitivity.csv")
    else:
        comparison = load_csv("five_day_model_comparison.csv")
        dm = load_csv("five_day_dm_sensitivity.csv")

    if not comparison.empty:
        st.subheader(f"{horizon} metrics")
        metric = st.selectbox("Metric", ["qlike", "mse"])
        if metric in comparison.columns:
            fig = px.bar(
                comparison.sort_values(metric),
                x="model",
                y=metric,
                color="model",
                labels={metric: metric.upper(), "model": "Model"},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')
        show_metric_table(comparison, "Stored comparison report")
    else:
        show_missing("Comparison report not found.")

    if not dm.empty:
        st.subheader("Diebold–Mariano sensitivity")
        if "hac_lags" in dm.columns and "holm_p_value" in dm.columns:
            fig = px.line(
                dm,
                x="hac_lags",
                y="holm_p_value",
                color="comparison",
                markers=True,
                labels={
                    "hac_lags": "HAC lag",
                    "holm_p_value": "Holm-adjusted p-value",
                    "comparison": "Comparison",
                },
            )
            fig.add_hline(
                y=0.05,
                line_dash="dash",
                annotation_text="0.05 reference",
            )
            st.plotly_chart(fig, width='stretch')

        show_metric_table(dm, "DM test results")

# ---------------------------------------------------------------------
# Volatility regimes
# ---------------------------------------------------------------------
elif page == "Volatility regimes":
    st.header("Model performance by volatility regime")

    horizon = st.radio(
        "Forecast horizon",
        ["1 day", "5 days"],
        horizontal=True,
    )

    filename = (
        "regime_comparison_1d.csv"
        if horizon == "1 day"
        else "regime_comparison_5d.csv"
    )
    regimes = load_csv(filename)

    if regimes.empty:
        show_missing("Regime comparison report not found.")
    else:
        metric = st.selectbox("Metric", ["qlike", "mse"])
        fig = px.bar(
            regimes,
            x="model",
            y=metric,
            color="regime",
            barmode="group",
            labels={
                "model": "Model",
                "regime": "Volatility regime",
                metric: metric.upper(),
            },
        )
        st.plotly_chart(fig, width='stretch')
        show_metric_table(regimes, "Regime comparison report")

# ---------------------------------------------------------------------
# Strategy backtests
# ---------------------------------------------------------------------
elif page == "Strategy backtests":
    st.header("Volatility-targeting strategy backtests")
    st.warning(
        "Historical backtests are development-period results. They are not "
        "a guarantee of future performance."
    )

    strategy_files = {
        "EWMA": "strategy_ewma.parquet",
        "GARCH": "strategy_garch.parquet",
        "GJR-GARCH": "strategy_gjr_garch.parquet",
        "HAR-RV": "strategy_har_rv.parquet",
        "LightGBM": "strategy_lightgbm.parquet",
    }

    available = {
        model: load_parquet(filename)
        for model, filename in strategy_files.items()
    }
    available = {
        model: df for model, df in available.items() if not df.empty
    }

    if not available:
        show_missing("No strategy backtest artifacts were found.")
    else:
        selected = st.multiselect(
            "Strategies",
            options=list(available),
            default=list(available),
        )

        curves = []
        summaries = []

        for model in selected:
            df = available[model].copy()
            required = {"next_date", "net_return", "gross_return"}
            if not required.issubset(df.columns):
                st.warning(f"{model}: expected return columns are missing.")
                continue

            df["next_date"] = pd.to_datetime(df["next_date"])
            df = df.sort_values("next_date")
            df = df.dropna(subset=["net_return", "gross_return"])

            if df.empty:
                continue

            df["net_growth"] = (1 + df["net_return"]).cumprod()
            df["gross_growth"] = (1 + df["gross_return"]).cumprod()
            df["model"] = model
            curves.append(df)

            net = df["net_return"]
            wealth = df["net_growth"]
            drawdown = wealth / wealth.cummax() - 1

            summaries.append(
                {
                    "model": model,
                    "observations": len(df),
                    "cumulative_net_return": wealth.iloc[-1] - 1,
                    "cumulative_gross_return": df["gross_growth"].iloc[-1] - 1,
                    "annualized_net_volatility": (
                        net.std(ddof=1) * np.sqrt(252)
                        if len(net) > 1
                        else np.nan
                    ),
                    "maximum_drawdown": drawdown.min(),
                    "total_transaction_cost": df["transaction_cost"].sum()
                    if "transaction_cost" in df.columns
                    else np.nan,
                    "average_turnover": df["turnover"].mean()
                    if "turnover" in df.columns
                    else np.nan,
                }
            )

        if curves:
            curve_data = pd.concat(curves, ignore_index=True)
            st.subheader("Cumulative strategy growth")
            fig = px.line(
                curve_data,
                x="next_date",
                y="net_growth",
                color="model",
                labels={
                    "next_date": "Date",
                    "net_growth": "Growth of 1 unit",
                    "model": "Strategy",
                },
            )
            st.plotly_chart(fig, width='stretch')

            summary = pd.DataFrame(summaries)
            st.subheader("Backtest summary")
            st.dataframe(
                summary.style.format(
                    {
                        "cumulative_net_return": "{:.2%}",
                        "cumulative_gross_return": "{:.2%}",
                        "annualized_net_volatility": "{:.2%}",
                        "maximum_drawdown": "{:.2%}",
                        "total_transaction_cost": "{:.4f}",
                        "average_turnover": "{:.4f}",
                    }
                ),
                width='stretch',
                hide_index=True,
            )

            with st.expander("Inspect selected strategy records"):
                st.dataframe(curve_data, width='stretch', hide_index=True)

st.sidebar.divider()
st.sidebar.caption(
    "VolatilityForge | Research dashboard\n\n"
    "Educational use only. Historical results do not guarantee future performance."
)
