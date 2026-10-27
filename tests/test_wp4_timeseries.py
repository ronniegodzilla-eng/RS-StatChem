"""WP4 forecasting: ACF/PACF/CCF, seasonal decomposition, ETS/ARIMA models (Box-Jenkins airline
series), Create Time Series and Define Dates."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa import stattools
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import timeseries as ts

DATA = Path(__file__).parent / "data_wp4"


def _table(section, title):
    return next(i for i in section.items if getattr(i, "title", "").startswith(title))


@pytest.fixture()
def air():
    a = pd.read_csv(DATA / "airpassengers.csv")
    ds = Dataset(pd.DataFrame({"pass": a["value"].astype(float)}))
    ts.define_dates(ds, "years_months", [1949, 1])
    return ds


def test_define_dates(air):
    assert ts.periodicity(air) == 12
    assert air.df["DATE_"].iloc[13] == "FEB 1950"
    assert air.df["YEAR_"].iloc[-1] == 1960 and air.df["MONTH_"].iloc[-1] == 12
    ds = Dataset(pd.DataFrame({"y": np.arange(10.0)}))
    ts.define_dates(ds, "years_quarters", [2020, 3])
    assert list(ds.df["DATE_"][:3]) == ["Q3 2020", "Q4 2020", "Q1 2021"]


def test_acf_pacf_ljung_box_match_statsmodels(air):
    y = np.log(air.df["pass"].to_numpy())
    z = np.diff(y)
    k, r, se, q, p = ts.acf_table(z, 16)
    np.testing.assert_allclose(r, stattools.acf(z, nlags=16, fft=False)[1:], rtol=1e-10)
    lb = acorr_ljungbox(z, lags=16)
    np.testing.assert_allclose(q, lb["lb_stat"], rtol=1e-10)
    np.testing.assert_allclose(p, lb["lb_pvalue"], rtol=1e-8)
    n = len(z)
    np.testing.assert_allclose(se, np.sqrt((n - k) / (n * (n + 2))))
    np.testing.assert_allclose(ts.pacf_values(z, 16), stattools.pacf(z, nlags=16,
                                                                     method="ldb")[1:],
                               rtol=1e-8)
    _, _, se_b, _, _ = ts.acf_table(z, 5, "bartlett")
    assert se_b[0] == pytest.approx(1 / np.sqrt(n))
    assert se_b[2] == pytest.approx(np.sqrt((1 + 2 * (r[0] ** 2 + r[1] ** 2)) / n))


def test_ccf_definition():
    rng = np.random.default_rng(0)
    x = rng.normal(size=80)
    y = np.r_[0, 0, x[:-2]] + 0.3 * rng.normal(size=80)   # y lags x by 2
    lags, r, se = ts.ccf_values(x, y, 5)
    assert lags[np.argmax(r)] == 2
    xc, yc = x - x.mean(), y - y.mean()
    ref = np.sum(xc[:-2] * yc[2:]) / 80 / (x.std() * y.std())
    assert r[lags == 2][0] == pytest.approx(ref)
    assert se[lags == 0][0] == pytest.approx(1 / np.sqrt(80))


def test_seasonal_factors_match_statsmodels(air):
    x = air.df["pass"].to_numpy()
    for model, sm_model in (("multiplicative", "multiplicative"), ("additive", "additive")):
        res = ts.seasonal_decompose(x, 12, model)
        ref = seasonal_decompose(x, model=sm_model, period=12)
        np.testing.assert_allclose(res["SAF"], ref.seasonal, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(res["MA"][6:-6], ref.trend[6:-6])
    out = ts.seasonal_decomposition(air, ["pass"])
    assert set(out.saved) == {"ERR", "SAS", "SAF", "STC"}
    rows = _table(out.section, "Seasonal Factors").rows
    assert rows[6][1] == "122.7"   # July


def test_airline_model_matches_box_jenkins_r_fixture(air):
    """R: arima(log(AirPassengers), c(0,1,1), seasonal=c(0,1,1)) -> ma1 = -0.4018 (se
    0.0896), sma1 = -0.5569 (se 0.0731), loglik 244.7. SPSS prints MA with the opposite
    sign (0.402, 0.557)."""
    spec = ts.ModelSpec("arima", order=(0, 1, 1), seasonal=(0, 1, 1), transform="log",
                        constant=False)
    y = air.df["pass"].to_numpy()
    fm = ts.fit_model(y, spec, 12, horizon=12)
    params = {p[0]: p for p in fm.params}
    # the likelihood is flat near the optimum: optimisers/BLAS builds differ by up to ~1e-3
    # (CI's macOS runner gave 0.5562), far inside the standard errors of 0.07-0.09
    assert params["MA Lag 1"][1] == pytest.approx(0.4018, abs=2e-3)
    assert params["MA, Seasonal Lag 1"][1] == pytest.approx(0.5569, abs=2e-3)
    assert params["MA Lag 1"][2] == pytest.approx(0.0896, abs=2e-3)
    assert params["MA, Seasonal Lag 1"][2] == pytest.approx(0.0731, abs=2e-3)
    import statsmodels.api as sm

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sm.tsa.SARIMAX(np.log(y), order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(
            disp=0)
    assert ref.llf == pytest.approx(244.7, abs=0.05)
    np.testing.assert_allclose(fm.forecast, np.exp(ref.forecast(12)), rtol=2e-3)
    sec = ts.forecast_models(air, ["pass"], "arima", spec, horizon=12, residual_acf=True)
    stats_row = _table(sec, "Model Statistics").rows[0]
    assert float(stats_row[3]) > 0.98            # R-squared
    fc = _table(sec, "Forecast")
    assert fc.columns[1] == "JAN 1961" and len(fc.rows) == 3


def test_ets_models_match_statsmodels(air):
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    y = air.df["pass"].to_numpy()
    fm = ts.fit_model(y, ts.ModelSpec("ets", "simple"), 12, horizon=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = ETSModel(pd.Series(y), error="add").fit(disp=False)
    assert fm.params[0][1] == pytest.approx(ref.params[0], abs=1e-4)
    np.testing.assert_allclose(fm.forecast, ref.forecast(4), rtol=1e-6)
    for kind in ("holt", "brown", "damped", "simple_seasonal", "winters_add", "winters_mult"):
        fm = ts.fit_model(y, ts.ModelSpec("ets", kind), 12, horizon=6)
        assert len(fm.forecast) == 6 and np.all(fm.lower < fm.forecast)
        assert np.all(fm.forecast < fm.upper)
        assert 0.8 < fm.stats["R-squared"] <= 1
    # Holt on this series: the likelihood is multimodal; the best solution is alpha ~ 1
    holt = ts.fit_model(y, ts.ModelSpec("ets", "holt"), 12)
    assert holt.params[0][1] > 0.9


def test_fit_statistics_definitions():
    rng = np.random.default_rng(3)
    y = 50 + np.cumsum(rng.normal(size=60))
    fm = ts.fit_model(y, ts.ModelSpec("arima", order=(1, 0, 0)), 1)
    ok = ~np.isnan(fm.fitted)
    e = y[ok] - fm.fitted[ok]
    assert fm.stats["RMSE"] == pytest.approx(np.sqrt(np.mean(e ** 2)))
    assert fm.stats["MAE"] == pytest.approx(np.mean(np.abs(e)))
    assert fm.stats["MAPE"] == pytest.approx(np.mean(np.abs(e / y[ok])) * 100)
    k = fm.n_params
    assert fm.stats["Normalized BIC"] == pytest.approx(
        np.log(np.mean(e ** 2)) + (k + 1) * np.log(ok.sum()) / ok.sum())


def test_expert_modeler_and_save_extends_cases(air):
    n0 = air.n_cases
    sec = ts.forecast_models(air, ["pass"], "expert", horizon=6,
                             save=("predicted", "lcl", "ucl", "nresidual"), store=True)
    assert air.n_cases == n0 + 6
    assert air.df["DATE_"].iloc[-1] == "JUN 1961"
    pred = air.df["Predicted_pass_Model_1"]
    assert pred.notna().sum() >= n0
    assert air.df["LCL_pass_Model_1"].notna().sum() == 6
    label = _table(sec, "Model Description").rows[0][2]
    assert "Simple" not in label            # a seasonal model is chosen for this series
    stored = air.meta["forecast_models"]["pass"]
    again = ts.forecast_models(air, ["pass"], "stored", horizon=3, fixed={"pass": stored})
    assert _table(again, "Forecast").rows[0][1]


def test_arima_with_predictor():
    rng = np.random.default_rng(8)
    n = 100
    x = rng.normal(size=n + 5)
    e = np.zeros(n + 5)
    for t in range(1, n + 5):
        e[t] = 0.5 * e[t - 1] + rng.normal()
    y = 10 + 3 * x + e
    ds = Dataset(pd.DataFrame({"y": np.r_[y[:n], [np.nan] * 5], "x": x}))
    sec = ts.forecast_models(ds, ["y"], "arima", ts.ModelSpec("arima", order=(1, 0, 0)),
                             predictors=["x"], horizon=5)
    rows = {r[1]: r for r in _table(sec, "ARIMA Model Parameters").rows}
    assert float(rows["x Numerator Lag 0"][2]) == pytest.approx(3.0, abs=0.3)
    assert float(rows["AR Lag 1"][2]) == pytest.approx(0.5, abs=0.2)
    assert len(_table(sec, "Forecast").rows[0]) == 6


def test_create_time_series_functions():
    x = np.array([3.0, 5, 4, 8, 10, 9, 12, 15, 14, 18, 20, 19, 25])
    s = pd.Series(x)
    np.testing.assert_allclose(ts.series_function(x, "DIFF", 2)[2:], s.diff().diff()[2:])
    np.testing.assert_allclose(ts.series_function(x, "SDIFF", 1, period=4)[4:],
                               (s - s.shift(4))[4:])
    np.testing.assert_allclose(ts.series_function(x, "PMA", span=3)[3:],
                               s.rolling(3).mean().shift(1)[3:])
    np.testing.assert_allclose(ts.series_function(x, "MA", span=3)[1:-1],
                               s.rolling(3, center=True).mean()[1:-1])
    ma4 = ts.series_function(x, "MA", span=4)
    assert ma4[2] == pytest.approx((0.5 * 3 + 5 + 4 + 8 + 0.5 * 10) / 4)
    np.testing.assert_allclose(ts.series_function(x, "RMED", span=3)[1:-1],
                               s.rolling(3, center=True).median()[1:-1])
    np.testing.assert_allclose(ts.series_function(x, "CSUM"), np.cumsum(x))
    np.testing.assert_allclose(ts.series_function(x, "LAG", 2)[2:], x[:-2])
    np.testing.assert_allclose(ts.series_function(x, "LEAD", 1)[:-1], x[1:])
    sm = ts.series_function(x, "T4253H")
    assert np.all(np.isfinite(sm)) and np.std(np.diff(sm)) < np.std(np.diff(x))
    ds = Dataset(pd.DataFrame({"x": x}))
    sec = ts.create_time_series(ds, [("x_1", "DIFF", "x", 1, 3, 12)])
    assert _table(sec, "Created Series").rows[0][1:4] == ["2", "13", "12"]
    assert ds.var("x_1").label == "DIFF(x 1)"


def test_sequence_autocorrelation_spectral_sections(air):
    sec = ts.sequence_chart(air, ["pass"], log=True, diff=1)
    assert any(i.kind == "image" for i in sec.items)
    sec = ts.autocorrelations(air, ["pass"], 24, log=True, diff=1, sdiff=1, periodic_only=True)
    rows = _table(sec, "Autocorrelations").rows
    assert [r[0] for r in rows] == ["12", "24"]
    sec = ts.spectral(air, ["pass"], "daniell", 3)
    per = _table(sec, "Univariate Statistics").rows
    x = air.df["pass"].to_numpy()
    x = x - x.mean()
    f = np.fft.rfft(x)
    ref = 2 * np.abs(f[12]) ** 2 / len(x)          # periodogram n/2 (a^2 + b^2) at k = 12
    assert float(per[12][5]) == pytest.approx(ref, rel=1e-3)
    assert float(per[12][2]) == pytest.approx(12.0)  # the yearly cycle
