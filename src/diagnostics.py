from __future__ import annotations
import numpy as np
from src.core import acf


def ljung_box(residuals, nlags=10, fit_df=0):
    """Ljung-Box Q-test for residual autocorrelation.

    H0: The residuals are independently distributed (white noise).
    A p-value below 0.05 suggests the model has not captured all
    temporal structure.

    Statistic: Q = n(n+2) * sum(r_k^2 / (n-k)) for k=1..nlags
    Distribution: chi-squared(nlags - fit_df)

    The degrees of freedom are reduced by the number of estimated
    model parameters (p + q + P + Q for SARIMA) to avoid loss of
    power.  Reference: Ljung & Box (1978) Biometrika; Box, Jenkins,
    Reinsel 'Time Series Analysis' (4th ed.), Sec 8.3.
    """
    n = len(residuals)
    r = acf(residuals, nlags)
    q_stat = n * (n + 2) * np.sum(r[1:] ** 2 / (n - np.arange(1, nlags + 1) + 1e-10))
    df = max(1, nlags - fit_df)
    try:
        from scipy.stats import chi2
        p_val = 1.0 - chi2.cdf(q_stat, df)
    except ImportError:
        p_val = None
    return q_stat, p_val


def adf_test(series, max_lag=10):
    """Augmented Dickey-Fuller test for unit root (stationarity).

    H0: The series has a unit root (non-stationary).
    If the test statistic is less than the 5% critical value,
    the null is rejected and the series is considered stationary.

    Equation:  Delta y_t = alpha + beta * t + gamma * y_{t-1}
                            + sum(delta_i * Delta y_{t-i}) + eps_t

    The coefficient of interest is gamma on the lagged level.
    gamma = 0 under H0; gamma < 0 under H1 (stationary).

    Critical values are the MacKinnon (1994) finite-sample values.

    Reference: Hamilton 'Time Series Analysis', Ch. 17.
               Dickey & Fuller (1979) JASA 74(366).
    """
    y = np.asarray(series, float)
    n = len(y)
    dy = np.diff(y)
    y_lag = y[:-1]

    best_aic = np.inf
    best_t_stat = None
    best_p = 0
    best_n_obs = 0

    for p in range(max_lag + 1):
        X_list = []
        y_list = []
        for t in range(p + 1, len(dy)):
            row = [1.0, float(t)]
            row.append(float(y_lag[t]))
            for i in range(1, p + 1):
                row.append(float(dy[t - i]) if t - i >= 0 else 0.0)
            X_list.append(row)
            y_list.append(dy[t])

        X = np.array(X_list)
        y_reg = np.array(y_list)
        n_reg = len(y_reg)
        if n_reg < p + 3:
            continue

        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
        resid = y_reg - X @ beta
        sigma2 = np.sum(resid ** 2) / n_reg
        aic = n_reg * np.log(sigma2) + 2.0 * len(beta)

        if aic < best_aic:
            best_aic = aic
            best_p = p
            best_n_obs = n_reg
            try:
                se = np.sqrt(sigma2 * np.diag(np.linalg.inv(X.T @ X + 1e-10 * np.eye(X.shape[1]))))
                best_t_stat = float(beta[2] / (se[2] + 1e-10))
            except np.linalg.LinAlgError:
                best_t_stat = -1.0

    # MacKinnon (2010) critical values for Case 4: constant + linear trend.
    # The regression estimates:  y_t = alpha + beta * t + gamma * y_{t-1}
    #   + sum(delta_i * Diff y_{t-i}) + eps_t.
    # Null: gamma = 0 (unit root).  Values are for n ~ 500.
    # Reference: MacKinnon (2010) "Critical Values for Cointegration Tests"
    #   — Working Paper 1227, Queen's Economics Department.
    cv = {1: -3.96, 5: -3.41, 10: -3.13}

    if best_t_stat is None:
        return {"t_stat": None, "is_stationary": False, "critical_values": cv, "lag": 0, "n_obs": 0}

    return {
        "t_stat": round(best_t_stat, 4),
        "is_stationary": best_t_stat < cv[5],
        "critical_values": {str(k): round(v, 3) for k, v in cv.items()},
        "lag": best_p,
        "n_obs": best_n_obs,
    }
