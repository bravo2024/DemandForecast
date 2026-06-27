from __future__ import annotations

import numpy as np
import pytest

from src.data import make_synthetic, detect_frequency
from src.core import smape, mase, rmse, mae, acf, pacf
from src.diagnostics import ljung_box, adf_test
from src.models import (
    SeasonalNaive,
    RidgeForecaster,
    HoltWinters,
    RandomForestRegressor,
    ARIMAForecaster,
    SARIMAForecaster,
    GARCHResiduals,
)
from src.validation import walk_forward


def test_data_shape():
    s = make_synthetic(100)
    assert len(s) == 100
    assert s.dtype == float


def test_detect_frequency():
    n = 200
    s = make_synthetic(n)
    f = detect_frequency(s)
    assert isinstance(f, str)
    assert f in ("daily", "monthly", "unknown")


def test_seasonal_naive():
    s = make_synthetic(200)
    m = SeasonalNaive(7)
    m.fit(s, 7)
    p = m.predict(7)
    assert len(p) == 7
    assert np.all(p >= 0)


def test_ridge():
    s = make_synthetic(200)
    m = RidgeForecaster()
    m.fit(s, 7)
    p = m.predict(7)
    assert len(p) == 7
    assert np.all(p >= 0)


def test_hw():
    s = make_synthetic(200)
    m = HoltWinters(7)
    m.fit(s, 7)
    p = m.predict(7)
    assert len(p) == 7


def test_rf():
    s = make_synthetic(200)
    m = RandomForestRegressor(n_trees=5, max_depth=4)
    m.fit(s, 7)
    p = m.predict(7)
    assert len(p) == 7
    assert np.all(p >= 0)


def test_arima():
    s = make_synthetic(300)
    m = ARIMAForecaster(order=(1, 1, 1))
    m.fit(s, 7)
    p = m.predict(7)
    assert len(p) == 7
    assert np.all(np.isfinite(p))


def test_sarima():
    s = make_synthetic(400)
    m = SARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 0, 1, 7))
    m.fit(s, 7)
    p = m.predict(7)
    assert len(p) == 7
    assert np.all(np.isfinite(p))


def test_garch():
    np.random.seed(42)
    residuals = np.random.randn(200) * 2.0
    g = GARCHResiduals()
    g.fit(residuals)
    var = g.forecast_variance(10)
    assert len(var) == 10
    assert np.all(var > 0)


def test_acf():
    s = make_synthetic(200)
    v = acf(s, 20)
    assert len(v) == 21  # lag 0 .. 20
    assert abs(v[0] - 1.0) < 1e-10  # lag-0 ACF is always 1


def test_pacf():
    s = make_synthetic(200)
    v = pacf(s, 20)
    assert len(v) == 21
    assert abs(v[0] - 1.0) < 1e-10


def test_acf_flat():
    s = np.ones(100)
    v = acf(s, 10)
    assert np.allclose(v, 1.0, atol=1e-10)


def test_pacf_flat():
    s = np.ones(100)
    v = pacf(s, 10)
    assert v[0] == 1.0
    assert np.allclose(v[2:], 0.0, atol=1e-10)


def test_acf_nan_handling():
    s = np.array([np.nan, 1.0, 2.0, 3.0, np.nan, 5.0, 7.0, 9.0])
    v = acf(s, 3)
    assert len(v) == 4
    assert np.all(np.isfinite(v))


def test_ljung_box():
    np.random.seed(42)
    white_noise = np.random.randn(200)
    q, pv = ljung_box(white_noise, nlags=10)
    assert np.isfinite(q)
    assert isinstance(pv, float)


def test_adf():
    s = make_synthetic(200)
    result = adf_test(s)
    assert "t_stat" in result
    assert "is_stationary" in result
    assert "lag" in result
    assert "critical_values" in result


def test_walk_forward():
    s = make_synthetic(200)
    m = SeasonalNaive(7)
    errors = walk_forward(m, s, horizon=7, n_windows=3)
    assert len(errors) >= 1
    assert "rmse" in errors[0]


def test_smape():
    y = np.array([10, 20, 30])
    p = np.array([11, 19, 32])
    v = smape(y, p)
    assert v > 0 and v < 100


def test_mase():
    y = np.array([10, 20, 30])
    p = np.array([11, 19, 32])
    insample = np.array([8, 12, 18, 22, 28, 33])
    v = mase(y, p, insample, seasonality=1)
    assert np.isfinite(v)


@pytest.mark.skip(reason="Data fetchers require network access")
def test_fetch_fred():
    from src.data import fetch_fred
    series, dates = fetch_fred()
    assert len(series) > 10
    assert len(dates) == len(series)


@pytest.mark.skip(reason="Data fetchers require network access")
def test_fetch_wmt():
    from src.data import fetch_wmt
    series, dates = fetch_wmt()
    assert len(series) > 10
    assert len(dates) == len(series)


@pytest.mark.skip(reason="Data fetchers require network access")
def test_fetch_prophet():
    from src.data import fetch_prophet_retail
    series, dates = fetch_prophet_retail()
    assert len(series) > 10
    assert len(dates) == len(series)
