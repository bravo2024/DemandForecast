from __future__ import annotations
import numpy as np
from src.core import RidgeRegression, Standardizer
from src.features import FeatureEngineer


class SeasonalNaive:
    """Forecast equals the last observed value from the same weekday.
    This serves as the baseline that any competitive model should outperform."""

    def __init__(self, seasonality=7):
        self.seasonality = seasonality

    def fit(self, series, horizon=7):
        self._series = np.asarray(series, float)
        return self

    def predict(self, n_steps):
        s = self._series
        if len(s) < self.seasonality:
            return np.full(n_steps, np.mean(s))
        preds = []
        for i in range(n_steps):
            idx = -(self.seasonality - i % self.seasonality)
            if abs(idx) > len(s):
                idx = -self.seasonality
            preds.append(s[idx])
        return np.array(preds)


class RidgeForecaster:
    """Ridge regression with engineered lag, rolling, and calendar features.
    Offers interpretability and speed, making it a strong linear baseline."""

    def __init__(self, alpha=1.0, lags=(1, 2, 3, 7, 14, 28),
                 rolling_windows=(7, 28), use_fourier=True):
        self.alpha = alpha
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.use_fourier = use_fourier

    def fit(self, series, horizon=7):
        series = np.asarray(series, float)
        self._series = series.copy()
        self._fe = FeatureEngineer(
            lags=self.lags,
            rolling_windows=self.rolling_windows,
            add_fourier=self.use_fourier,
        )
        X, y = self._fe.fit(series).transform(series)
        self._scaler = Standardizer().fit(X)
        Xs = self._scaler.transform(X)
        self._model = RidgeRegression(alpha=self.alpha).fit(Xs, y)
        self._feat_names = self._fe.get_feature_names()
        return self

    def predict(self, n_steps):
        series = self._series.copy()
        preds = []
        for _ in range(n_steps):
            feats = self._fe.forecast_features(series)
            feats = self._scaler.transform(feats)
            p = self._model.predict(feats)[0]
            p = max(p, 0.0)
            preds.append(p)
            series = np.append(series, p)
        return np.array(preds)

    def feature_importance(self):
        coef = self._model.coef_raw_
        names = self._feat_names
        return sorted(zip(names, coef), key=lambda x: abs(x[1]), reverse=True)


class _RegressionTree:
    def __init__(self, max_depth=10, min_samples_leaf=5, max_features=None, seed=42):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed

    def fit(self, X, y):
        self._y = y.copy()
        rng = np.random.default_rng(self.seed)
        self.tree_ = self._grow(np.asarray(X, float), np.asarray(y, float), 0, rng)
        return self

    def _grow(self, X, y, depth, rng):
        node = {"value": float(np.mean(y)), "n": len(y)}
        if depth >= self.max_depth or len(y) < self.min_samples_leaf:
            return node

        parent_var = float(np.var(y)) if len(y) > 1 else 0.0
        best_gain, best_split = 0.0, None
        n_feat = X.shape[1]

        feat_idx = (
            rng.choice(n_feat, min(self.max_features, n_feat), replace=False)
            if self.max_features and self.max_features < n_feat
            else np.arange(n_feat)
        )

        for f in feat_idx:
            col = X[:, f]
            candidates = np.unique(col)
            if len(candidates) > 40:
                candidates = np.percentile(col, np.linspace(10, 90, 20))
            for thresh in candidates:
                left = col <= thresh
                nl, nr = int(left.sum()), int((~left).sum())
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                var_l = float(np.var(y[left])) if nl > 1 else 0.0
                var_r = float(np.var(y[~left])) if nr > 1 else 0.0
                gain = parent_var - (nl / len(y) * var_l + nr / len(y) * var_r)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (f, thresh)

        if best_split is None or best_gain < 1e-12:
            return node

        f, thresh = best_split
        left = X[:, f] <= thresh
        node["feature"] = int(f)
        node["threshold"] = float(thresh)
        node["left"] = self._grow(X[left], y[left], depth + 1, rng)
        node["right"] = self._grow(X[~left], y[~left], depth + 1, rng)
        return node

    def predict(self, X):
        X = np.asarray(X, float)
        return np.array([self._predict_row(x, self.tree_) for x in X])

    def _predict_row(self, x, node):
        if "feature" not in node:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_row(x, node["left"])
        return self._predict_row(x, node["right"])


class RandomForestRegressor:
    """Pure NumPy random forest for regression. Uses variance-reduction splits
    and mean leaf predictions. Slower than sklearn but exposes the underlying
    mechanics of bootstrapping, feature subsampling, and ensemble averaging."""

    def __init__(self, n_trees=30, max_depth=8, min_samples_leaf=5,
                 max_features="sqrt", seed=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed

    def fit(self, series, horizon=7):
        series = np.asarray(series, float)
        self._series = series.copy()
        self._fe = FeatureEngineer(
            lags=(1, 2, 3, 7, 14, 28),
            rolling_windows=(7, 28),
            add_fourier=True,
        )
        X, y = self._fe.fit(series).transform(series)
        self._scaler = Standardizer().fit(X)
        Xs = self._scaler.transform(X)
        self._y_store = y.copy()

        rng = np.random.default_rng(self.seed)
        n = len(Xs)
        mtry = int(np.sqrt(Xs.shape[1])) if self.max_features == "sqrt" else Xs.shape[1]

        self.trees_ = []
        oob_sums = np.zeros(n)
        oob_counts = np.zeros(n)

        for i in range(self.n_trees):
            seed_i = rng.integers(0, 2 ** 31)
            trng = np.random.default_rng(seed_i)
            idx = trng.integers(0, n, n)
            oob = np.setdiff1d(np.arange(n), np.unique(idx))

            tree = _RegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=mtry,
                seed=int(seed_i),
            )
            tree.fit(Xs[idx], y[idx])
            self.trees_.append(tree)

            if len(oob) > 0:
                oob_sums[oob] += tree.predict(Xs[oob])
                oob_counts[oob] += 1

        mask = oob_counts > 0
        self.oob_preds_ = np.full(n, np.nan)
        self.oob_preds_[mask] = oob_sums[mask] / oob_counts[mask]
        return self

    def predict(self, n_steps):
        series = self._series.copy()
        preds = []
        for _ in range(n_steps):
            feats = self._fe.forecast_features(series)
            feats = self._scaler.transform(feats)
            forest_preds = np.array([t.predict(feats)[0] for t in self.trees_])
            p = float(np.mean(forest_preds))
            p = max(p, 0.0)
            preds.append(p)
            series = np.append(series, p)
        return np.array(preds)

    def oob_score(self):
        if not hasattr(self, "oob_preds_") or self.oob_preds_ is None:
            return None
        y = self._y_store
        mask = ~np.isnan(self.oob_preds_)
        if mask.sum() < 2:
            return float("nan")
        ss_res = np.sum((y[mask] - self.oob_preds_[mask]) ** 2)
        ss_tot = np.sum((y[mask] - np.mean(y[mask])) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


class HoltWinters:
    """Additive Holt-Winters with level, trend, and seasonal components.
    This classic state-space method uses alpha (level), beta (trend), and
    gamma (seasonal) to control adaptation speed. Values near 1 produce
    fast adaptation with noisier estimates; values near 0 yield smoother,
    slower-changing estimates.

    Reference: Holt (1957), Winters (1960), Hyndman & Athanasopoulos
    'Forecasting: Principles and Practice' (3rd ed.), Ch. 8."""

    def __init__(self, seasonality=7, alpha=0.3, beta=0.1, gamma=0.2):
        self.seasonality = seasonality
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def fit(self, series, horizon=7):
        series = np.asarray(series, float)
        n, m = len(series), self.seasonality

        if n < 2 * m:
            self._components = {"level": float(np.mean(series)), "trend": 0.0,
                                "seasonal": np.zeros(m)}
            self._fitted = np.full(n, np.mean(series))
            return self

        seasonal = np.array([series[i] - np.mean(series[:m]) for i in range(m)])
        seasonal -= seasonal.mean()
        level = float(np.mean(series[:m]))
        trend = float((np.mean(series[m:2 * m]) - np.mean(series[:m])) / m)

        self._fitted = np.full(n, np.nan)
        for i in range(n):
            if i < m:
                self._fitted[i] = level + seasonal[i % m]
                continue
            prev_level = level
            prev_trend = trend
            level = self.alpha * (series[i] - seasonal[i % m]) + (1 - self.alpha) * (prev_level + prev_trend)
            trend = self.beta * (level - prev_level) + (1 - self.beta) * prev_trend
            seasonal[i % m] = self.gamma * (series[i] - level) + (1 - self.gamma) * seasonal[i % m]
            self._fitted[i] = prev_level + prev_trend + seasonal[i % m]

        self._components = {"level": level, "trend": trend, "seasonal": seasonal.copy()}
        return self

    def predict(self, n_steps):
        if not hasattr(self, "_components"):
            return np.full(n_steps, 0.0)
        c = self._components
        start_idx = len(self._fitted)
        preds = []
        for i in range(n_steps):
            p = c["level"] + (i + 1) * c["trend"] + c["seasonal"][(start_idx + i) % self.seasonality]
            preds.append(max(p, 0.0))
        return np.array(preds)


class ARIMAForecaster:
    """ARIMA(p,d,q) model wrapping statsmodels. AR terms capture
    autoregressive structure; MA terms capture shock persistence;
    d differencing passes handle non-stationarity.

    The model equation (with backshift operator B):
    (1 - phi_1 B - ... - phi_p B^p) (1-B)^d y_t = c + (1 + theta_1 B + ... + theta_q B^q) eps_t

    Reference: Box, Jenkins, Reinsel 'Time Series Analysis' (4th ed.).
               Hyndman & Athanasopoulos 'Forecasting: Principles and Practice' (3rd ed.), Ch. 9."""

    def __init__(self, order=(1, 1, 1), use_auto=False):
        self.order = order
        self.use_auto = use_auto

    def fit(self, series, horizon=7):
        from statsmodels.tsa.arima.model import ARIMA as _ARIMA
        self._series = np.asarray(series, float)
        try:
            self._model = _ARIMA(self._series, order=self.order).fit(method_kwargs={"disp": False})
            self._fitted = True
        except Exception:
            self._fitted = False
        return self

    def predict(self, n_steps):
        if not self._fitted:
            return np.full(n_steps, np.mean(self._series[-10:]) if len(self._series) >= 10 else 0.0)
        raw = self._model.forecast(n_steps)
        preds = np.asarray(raw, float).flatten()
        return np.maximum(preds, 0.0)

    def summary(self):
        if not self._fitted:
            return "Model did not converge"
        return str(self._model.summary())


class SARIMAForecaster:
    """SARIMA(p,d,q)(P,D,Q,s) model wrapping statsmodels.
    Adds seasonal AR, differencing, and MA terms to ARIMA.
    The seasonal period s=7 captures weekly cycles in daily data.

    Equation: Phi(B) Phi_s(B^s) (1-B)^d (1-B^s)^D y_t = Theta(B) Theta_s(B^s) eps_t

    Reference: Box, Jenkins, Reinsel Ch. 9.
               Hyndman & Athanasopoulos Ch. 9."""

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, series, horizon=7):
        import statsmodels.api as sm
        self._series = np.asarray(series, float)
        try:
            self._model = sm.tsa.SARIMAX(
                self._series, order=self.order, seasonal_order=self.seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False
            ).fit(disp=False)
            self._fitted = True
        except Exception:
            self._fitted = False
        return self

    def predict(self, n_steps):
        if not self._fitted:
            return np.full(n_steps, np.mean(self._series[-10:]) if len(self._series) >= 10 else 0.0)
        raw = self._model.forecast(n_steps)
        preds = np.asarray(raw, float).flatten()
        return np.maximum(preds, 0.0)

    def summary(self):
        if not self._fitted:
            return "Model did not converge"
        return str(self._model.summary())


class GARCHResiduals:
    """GARCH(1,1) model for residual volatility. Fits to prediction errors
    and models time-varying variance: sigma^2_t = omega + alpha * eps^2_{t-1}
    + beta * sigma^2_{t-1}. Used to adjust prediction interval width.

    Reference: Engle (1982) 'Autoregressive Conditional Heteroscedasticity'.
               Bollerslev (1986) 'Generalized Autoregressive Conditional Heteroscedasticity'.
               Tsay 'Analysis of Financial Time Series' (3rd ed.), Ch. 3."""

    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q
        self._fitted = False

    def fit(self, residuals):
        from arch import arch_model
        try:
            self._model = arch_model(np.asarray(residuals, float) * 100,
                                     vol="Garch", p=self.p, q=self.q).fit(disp="off")
            self._fitted = True
        except Exception:
            self._fitted = False
        return self

    def forecast_variance(self, n_steps):
        if not self._fitted:
            return np.ones(n_steps)
        raw = self._model.forecast(horizon=n_steps)
        var = np.asarray(raw.variance.values[-1], float).flatten()
        return var / 10000.0  # scale back from percentage-squared

    def summary(self):
        if not self._fitted:
            return "GARCH did not converge"
        return str(self._model.summary())


def compare_models(series, horizon=7, n_windows=5, selected_models=None):
    """Runs walk-forward validation for each selected model and returns
    error summaries (RMSE, MAE, sMAPE) for comparison."""
    from src.validation import walk_forward
    from src.core import rmse, mae, smape

    defaults = {
        "SeasonalNaive": SeasonalNaive(seasonality=7),
        "Ridge": RidgeForecaster(),
        "RandomForest": RandomForestRegressor(n_trees=20),
        "HoltWinters": HoltWinters(seasonality=7),
    }
    models = selected_models if selected_models else defaults

    results = {}
    for name, model in models.items():
        errors = walk_forward(model, series, horizon, n_windows)
        valid = [e for e in errors if not np.isnan(e.get("rmse", np.nan))]
        if valid:
            results[name] = {
                "rmse": float(np.mean([e["rmse"] for e in valid])),
                "mae": float(np.mean([e["mae"] for e in valid])),
                "smape": float(np.mean([e["smape"] for e in valid])),
                "errors": errors,
            }
        else:
            results[name] = {"rmse": float("nan"), "mae": float("nan"),
                             "smape": float("nan"), "errors": errors}
    return results
