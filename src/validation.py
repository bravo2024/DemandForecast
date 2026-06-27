from __future__ import annotations
import copy
import numpy as np
from src.core import rmse, mae, smape


def walk_forward(model, series, horizon=7, n_windows=5):
    """Walk-forward validation on a single model instance.

    Each fold trains on an expanding window and forecasts the next
    horizon steps. This procedure simulates a production retraining
    schedule and is the gold standard for time-series evaluation.

    Parameters
    ----------
    model : object with .fit(series, horizon) and .predict(n_steps)
    series : 1-D array
    horizon : int — number of steps per forecast
    n_windows : int — number of folds
    """
    series = np.asarray(series, float)
    n = len(series)
    step = horizon
    min_train = max(horizon * 3, 100)

    errors = []
    for w in range(n_windows):
        train_end = min_train + w * step
        test_end = train_end + horizon
        if test_end > n:
            break

        train = series[:train_end]
        test = series[train_end:test_end]

        m = copy.deepcopy(model)

        try:
            m.fit(train, horizon=horizon)
            pred = m.predict(len(test))

            if len(pred) > len(test):
                pred = pred[:len(test)]
            elif len(pred) < len(test):
                pred = np.pad(pred, (0, len(test) - len(pred)),
                              mode="constant",
                              constant_values=pred[-1] if len(pred) else 0.0)

            errors.append({
                "fold": w,
                "train_end": train_end,
                "rmse": rmse(test, pred),
                "mae": mae(test, pred),
                "smape": smape(test, pred),
                "pred": pred.tolist(),
                "actual": test.tolist(),
            })
        except Exception as exc:
            errors.append({
                "fold": w,
                "train_end": train_end,
                "rmse": float("nan"),
                "mae": float("nan"),
                "smape": float("nan"),
                "error": str(exc),
            })

    return errors


def evaluate_at_horizons(series, model, horizon=7, n_windows=5):
    """Decomposes MAE by forecast step (t+1, t+2, ..., t+H).
    Short horizons almost always outperform long ones; this quantifies the decay."""
    series = np.asarray(series, float)
    step = horizon
    min_train = max(horizon * 3, 100)
    n = len(series)

    step_errors = [[] for _ in range(horizon)]

    for w in range(n_windows):
        train_end = min_train + w * step
        test_end = train_end + horizon
        if test_end > n:
            break

        train = series[:train_end]
        test = series[train_end:test_end]

        m = copy.deepcopy(model)

        try:
            m.fit(train, horizon=horizon)
            pred = m.predict(len(test))
            for h in range(min(len(test), len(pred), horizon)):
                step_errors[h].append(abs(test[h] - pred[h]))
        except Exception:
            pass

    return {
        "horizon": list(range(1, horizon + 1)),
        "mae_by_horizon": [
            float(np.mean(e)) if e else float("nan") for e in step_errors
        ],
    }


def _signed_forecast_errors(model, series, horizon, n_windows):
    """Collect per-step signed forecast errors from walk-forward evaluation.

    Returns a matrix of shape (n_windows, horizon) where element (w, h)
    is actual[t+h] - forecast[t+h] for the w-th fold.  Unlike per-fold
    MAE, signed errors preserve bias information and produce asymmetric
    prediction intervals."""
    series = np.asarray(series, float)
    step = horizon
    min_train = max(horizon * 3, 100)
    n = len(series)

    errors = np.full((n_windows, horizon), np.nan)
    fold = 0
    for w in range(n_windows):
        train_end = min_train + w * step
        test_end = train_end + horizon
        if test_end > n:
            break

        train = series[:train_end]
        test = series[train_end:test_end]

        m = copy.deepcopy(model)

        try:
            m.fit(train, horizon=horizon)
            pred = m.predict(len(test))
            for h in range(min(len(pred), len(test), horizon)):
                errors[fold, h] = test[h] - pred[h]
            fold += 1
        except Exception:
            continue

    return errors[:fold]


def prediction_intervals(series, model, horizon=7, n_simulations=200, seed=None):
    """Bootstrap-based 90% prediction intervals from signed per-step errors.

    Resamples historical walk-forward forecast errors to estimate the
    uncertainty distribution at each horizon step.  Because the bootstrap
    uses *signed* residuals, the intervals are naturally asymmetric when
    the model is biased — unlike MAE-based approaches that force symmetry.

    Method (cf. Efron & Tibshirani 'An Introduction to the Bootstrap',
    Ch. 8; Hyndman & Athanasopoulos 'Forecasting: Principles and Practice'
    (3rd ed.), Sec 5.5):

        1. Collect e_{w,h} = actual[t+h] - forecast[t+h] over w walk-forward folds.
        2. Resample with replacement for each simulation s:
               e*_{s,h} = random draw from {e_{w,h}}_w
        3. Prediction interval:  [Q_5(forecast[h] + e*_{s,h}),
                                   Q_95(forecast[h] + e*_{s,h})]

    Parameters
    ----------
    seed : int or None
        If None, each call produces different intervals (default).
        Set a fixed integer for reproducible results.
    """
    n_windows = max(5, n_simulations // 20)
    errors = _signed_forecast_errors(model, series, horizon, n_windows)
    if errors.shape[0] < 2:
        return None

    rng = np.random.default_rng(seed)
    boot = np.zeros((n_simulations, horizon))
    for h in range(horizon):
        col = errors[:, h]
        valid = col[np.isfinite(col)]
        if len(valid) < 2:
            boot[:, h] = 0.0
        else:
            boot[:, h] = rng.choice(valid, size=n_simulations)

    m = model.__class__()
    if hasattr(model, "__dict__"):
        for k, v in model.__dict__.items():
            if not k.startswith("_"):
                setattr(m, k, v)
    m.fit(series, horizon)
    point = m.predict(horizon)

    intervals = []
    for h in range(horizon):
        errors_h = point[h] + boot[:, h]
        lower = float(np.percentile(errors_h, 5))
        upper = float(np.percentile(errors_h, 95))
        intervals.append({
            "horizon": h + 1,
            "forecast": float(point[h]),
            "lower_90": max(lower, 0.0),
            "upper_90": upper if upper >= lower else float(point[h]),
        })
    return intervals
