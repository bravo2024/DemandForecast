# DemandForecast

Retail demand forecasting with walk-forward validation, multiple model families, and prediction intervals.

## Models

- Seasonal Naive (baseline)
- Ridge regression with lag features, rolling windows, calendar dummies, and Fourier terms
- Random Forest (from scratch — CART trees with bagging)
- Holt-Winters exponential smoothing (additive level, trend, seasonal)
- ARIMA(p,d,q) — via statsmodels
- SARIMA(p,d,q)(P,D,Q,s) — via statsmodels
- GARCH-adjusted prediction intervals — via arch

## Diagnostics (from scratch)

- ACF and PACF via Yule-Walker equations
- Augmented Dickey-Fuller test with AIC lag selection and MacKinnon (2010) critical values
- Ljung-Box Q-test with model-adjusted degrees of freedom

## Data sources

- FRED RSFSDP (monthly US retail sales, 2020-2025)
- Yahoo Finance WMT (daily Walmart stock price, 2020-2024)
- Prophet example retail_sales (monthly, 1992-2016)
- Synthetic fallback with trend, weekly/yearly seasonality, promotions, and stockouts

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q -W ignore
streamlit run app.py
```

### CLI

```bash
python train.py --horizon 14 --models naive ridge hw arima
python train.py --models arima sarima --arima-p 2 --arima-d 1 --arima-q 2
```

## Dashboard

The app has four tabs: Data Explorer (series plot, summary stats, ADF test), Seasonality and ACF/PACF (decomposition, autocorrelation plots with LaTeX equations), Model Training (walk-forward comparison, horizon decay, per-fold errors), and Forecast and Uncertainty (point forecast with 90% bootstrap prediction intervals, residual diagnostics).

## Project structure

```
src/
  data.py          data loaders (synthetic + live)
  core.py          metrics (RMSE, MAE, sMAPE, MASE) + ACF/PACF
  diagnostics.py   Ljung-Box, ADF test
  features.py      feature engineering (lags, windows, calendar, Fourier)
  models.py        six model classes + GARCHResiduals
  validation.py    walk_forward, evaluate_at_horizons, prediction_intervals
  evaluate.py      metric persistence
  persist.py       model save/load
app.py             Streamlit entry point
train.py           CLI entry point
```

## Notes

ARIMA and SARIMA estimation uses statsmodels. A from-scratch ARIMA implementation requires non-linear optimisation (Kalman filter or MLE) and is outside this project's scope. The GARCH volatility adjustment on prediction intervals is applied post-hoc rather than integrated into each model's forecast variance.
