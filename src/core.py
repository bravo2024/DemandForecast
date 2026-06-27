from __future__ import annotations
import numpy as np

# ---------- forecast metrics ----------

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

def mape(y_true, y_pred):
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = np.abs(y) > 1e-8
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y[mask] - p[mask]) / y[mask]))) * 100

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (0-200 scale).
    Used in the M5 competition. Bounded, handles zero values better than MAPE."""
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y) + np.abs(p)
    mask = denom > 1e-8
    if not mask.any():
        return float("nan")
    return float(np.mean(2.0 * np.abs(y[mask] - p[mask]) / denom[mask]) * 100)

def mase(y_true, y_pred, y_insample, seasonality=1):
    """Mean Absolute Scaled Error. Scale-independent metric.
    Values below 1 indicate the model outperforms the seasonal-naive baseline.
    Primary metric in the M4 competition. References: Hyndman & Koehler (2006)."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    y_insample = np.asarray(y_insample, float)
    if seasonality > 1 and len(y_insample) > seasonality:
        naive_errors = np.abs(y_insample[seasonality:] - y_insample[:-seasonality])
    else:
        naive_errors = np.abs(np.diff(y_insample))
    naive_mae = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
    if naive_mae < 1e-10:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / naive_mae)

def rmsse(y_true, y_pred, y_insample, seasonality=1):
    """Root Mean Squared Scaled Error. Primary metric in the M5 competition.
    Scale-independent; values are ratios relative to a seasonal-naive baseline."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    y_insample = np.asarray(y_insample, float)
    if seasonality > 1 and len(y_insample) > seasonality:
        naive_errors = (y_insample[seasonality:] - y_insample[:-seasonality]) ** 2
    else:
        naive_errors = np.diff(y_insample) ** 2
    naive_mse = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
    if naive_mse < 1e-10:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / np.sqrt(naive_mse))

# ---------- scaling ----------

class Standardizer:
    def fit(self, X):
        X = np.asarray(X, float)
        self.mu_ = X.mean(0)
        self.sd_ = X.std(0, ddof=1) + 1e-8
        return self
    def transform(self, X):
        return (np.asarray(X, float) - self.mu_) / self.sd_
    def fit_transform(self, X):
        return self.fit(X).transform(X)

# ---------- baseline model ----------

class RidgeRegression:
    """Closed-form ridge regression. The L2 regularizer addresses
    collinearity among lag features. Alpha=0 yields OLS; higher values
    shrink coefficients toward zero. Reference: Hoerl & Kennard (1970)."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        Xb = np.hstack([np.ones((len(X), 1)), X])
        A = Xb.T @ Xb + self.alpha * np.eye(Xb.shape[1])
        A[0, 0] -= self.alpha
        self.coef_ = np.linalg.solve(A, Xb.T @ y)
        return self
    def predict(self, X):
        X = np.asarray(X, float)
        return np.hstack([np.ones((len(X), 1)), X]) @ self.coef_
    @property
    def intercept_(self):
        return self.coef_[0]
    @property
    def coef_raw_(self):
        return self.coef_[1:]

# ---------- autocorrelation (from scratch) ----------

def acf(series, nlags=40):
    """Autocorrelation function at lags 0 through nlags.
    rho(k) = Cov(y_t, y_{t-k}) / Var(y_t)
    Reference: Box, Jenkins, Reinsel 'Time Series Analysis' (4th ed.), Ch. 3."""
    s = np.asarray(series, float)
    s = s[np.isfinite(s)]
    if len(s) < nlags + 2:
        return np.full(nlags + 1, np.nan)
    n = len(s)
    mu = np.mean(s)
    c0 = np.sum((s - mu) ** 2)
    if c0 < 1e-10:
        return np.ones(nlags + 1)
    acfs = np.ones(nlags + 1)
    for k in range(1, nlags + 1):
        acfs[k] = np.sum((s[:-k] - mu) * (s[k:] - mu)) / c0
    return acfs

def pacf(series, nlags=40):
    """Partial autocorrelation function via Yule-Walker equations at each lag.
    rho_kk = phi_kk where phi_kk is the last coefficient of the AR(k) fit.
    Reference: Shumway & Stoffer 'Time Series Analysis and Its Applications' (4th ed.), Sec 3.4."""
    r = acf(series, nlags)
    pacfs = np.ones(nlags + 1)
    if np.any(np.abs(r[1:]) > 0.9999):
        import warnings
        warnings.warn("Near-perfect autocorrelation detected — "
                      "Yule-Walker matrix may be singular. "
                      "PACF values beyond lag 1 may be unreliable.")
    for k in range(1, nlags + 1):
        R = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                R[i, j] = r[abs(i - j)]
        try:
            phi = np.linalg.solve(R, r[1:k+1])
            pacfs[k] = float(phi[-1])
        except np.linalg.LinAlgError:
            pacfs[k] = 0.0
    return pacfs
