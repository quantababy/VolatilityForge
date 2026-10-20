import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from src.compare_five_day_forecasts import build_dm_report
from src.compare_forecasts import run_dm_sensitivity


def _synthetic_losses(
    columns: tuple[str, ...],
    *,
    seed: int = 42,
    n: int = 100,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2024-01-02", periods=n)
    common = rng.normal(0.0, 0.2, size=n)

    data = {}
    for position, column in enumerate(columns):
        noise = rng.normal(0.0, 0.1 + position * 0.01, size=n)
        data[column] = 1.0 + common + noise + position * 0.01

    return pd.DataFrame(data, index=index)


def test_one_day_dm_report_has_expected_lags_and_comparisons() -> None:
    losses = _synthetic_losses(
        ("EWMA", "GARCH(1,1)", "GJR-GARCH(1,1)", "HAR-RV", "LightGBM")
    )

    report = run_dm_sensitivity(losses)

    assert len(report) == 12
    assert set(report["hac_lags"]) == {5, 10, 22}
    assert set(report["forecast_horizon"]) == {1}
    assert report["comparison"].nunique() == 4
    assert report["n_observations"].eq(len(losses)).all()


def test_five_day_dm_report_has_expected_lags_and_comparisons() -> None:
    losses = _synthetic_losses(
        ("EWMA", "GARCH(1,1)", "GJR-GARCH(1,1)", "HAR-RV"),
        seed=43,
    )

    report = build_dm_report(losses)

    assert len(report) == 12
    assert set(report["hac_lags"]) == {4, 5, 10, 22}
    assert set(report["forecast_horizon"]) == {5}
    assert report["comparison"].nunique() == 3
    assert report["n_observations"].eq(len(losses)).all()


def test_holm_adjustment_is_applied_within_each_one_day_lag_family() -> None:
    losses = _synthetic_losses(
        ("EWMA", "GARCH(1,1)", "GJR-GARCH(1,1)", "HAR-RV", "LightGBM"),
        seed=44,
    )

    report = run_dm_sensitivity(losses)

    for _, family in report.groupby("hac_lags"):
        expected = multipletests(family["p_value"], method="holm")[1]
        np.testing.assert_allclose(
            family["holm_p_value"].to_numpy(),
            expected,
            rtol=1e-10,
            atol=1e-12,
        )


def test_holm_adjustment_is_applied_within_each_five_day_lag_family() -> None:
    losses = _synthetic_losses(
        ("EWMA", "GARCH(1,1)", "GJR-GARCH(1,1)", "HAR-RV"),
        seed=45,
    )

    report = build_dm_report(losses)

    for _, family in report.groupby("hac_lags"):
        expected = multipletests(family["p_value"], method="holm")[1]
        np.testing.assert_allclose(
            family["holm_p_value"].to_numpy(),
            expected,
            rtol=1e-10,
            atol=1e-12,
        )


def test_dm_reports_have_valid_p_values_and_date_ranges() -> None:
    one_day = run_dm_sensitivity(
        _synthetic_losses(
            ("EWMA", "GARCH(1,1)", "GJR-GARCH(1,1)", "HAR-RV", "LightGBM"),
            seed=46,
        )
    )
    five_day = build_dm_report(
        _synthetic_losses(
            ("EWMA", "GARCH(1,1)", "GJR-GARCH(1,1)", "HAR-RV"),
            seed=47,
        )
    )

    for report in (one_day, five_day):
        for column in ("p_value", "holm_p_value"):
            assert report[column].between(0.0, 1.0).all()

        assert report["start_date"].eq("2024-01-02").all()
        assert report["end_date"].eq(
            pd.bdate_range("2024-01-02", periods=100)[-1].date().isoformat()
        ).all()
