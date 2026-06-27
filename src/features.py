from __future__ import annotations
import numpy as np


class FeatureEngineer:
    """Transforms a univariate time series into a supervised-learning feature
    matrix using lags, rolling windows, calendar indicators, and Fourier terms.

    A core principle in demand forecasting: feature engineering often matters
    more than the choice of ML model. Lag features capture persistence, rolling
    statistics capture recent trends, and calendar terms capture periodicities
    that lags alone cannot resolve.
    """

    def __init__(self, lags=(1, 2, 3, 7, 14, 28),
                 rolling_windows=(7, 28),
                 rolling_aggs=("mean",),
                 add_calendar=True,
                 add_fourier=True):
        self.lags = list(lags) if lags else []
        self.rolling_windows = list(rolling_windows) if rolling_windows else []
        self.rolling_aggs = rolling_aggs
        self.add_calendar = add_calendar
        self.add_fourier = add_fourier
        self._fitted = False

    def fit(self, series, dates=None):
        series = np.asarray(series, float)
        lookback = 0
        if self.lags:
            lookback = max(lookback, max(self.lags))
        if self.rolling_windows:
            lookback = max(lookback, max(self.rolling_windows))
        self.min_lookback_ = lookback

        n = 0
        if self.lags:
            n += len(self.lags)
        if self.rolling_windows:
            n += len(self.rolling_windows) * len(self.rolling_aggs)
        if self.add_calendar:
            n += 3
        if self.add_fourier:
            n += 6
        self.n_features_ = n
        self._fitted = True
        return self

    def transform(self, series, dates=None):
        if not self._fitted:
            self.fit(series, dates)
        series = np.asarray(series, float)
        rows, targets = [], []
        start = self.min_lookback_
        for t in range(start, len(series) - 1):
            row = []
            for lag in self.lags:
                row.append(series[t - lag])
            for w in self.rolling_windows:
                window = series[t - w:t]
                if "mean" in self.rolling_aggs:
                    row.append(float(np.mean(window)))
                if "std" in self.rolling_aggs:
                    row.append(float(np.std(window)))
                if "min" in self.rolling_aggs:
                    row.append(float(np.min(window)))
                if "max" in self.rolling_aggs:
                    row.append(float(np.max(window)))
            if self.add_calendar:
                row.append(t % 7)
                row.append((t % 365) // 30)
                row.append((t % 365) // 90)
            if self.add_fourier:
                row.append(np.sin(2 * np.pi * t / 7))
                row.append(np.cos(2 * np.pi * t / 7))
                row.append(np.sin(2 * np.pi * t / 365.25))
                row.append(np.cos(2 * np.pi * t / 365.25))
                row.append(np.sin(4 * np.pi * t / 365.25))
                row.append(np.cos(4 * np.pi * t / 365.25))
            rows.append(row)
            targets.append(series[t + 1])
        return np.array(rows), np.array(targets)

    def forecast_features(self, series):
        """Produces the feature row for the next unknown time step."""
        t = len(series)
        row = []
        for lag in self.lags:
            row.append(series[t - lag])
        for w in self.rolling_windows:
            window = series[t - w:t]
            if "mean" in self.rolling_aggs:
                row.append(float(np.mean(window)))
            if "std" in self.rolling_aggs:
                row.append(float(np.std(window)))
            if "min" in self.rolling_aggs:
                row.append(float(np.min(window)))
            if "max" in self.rolling_aggs:
                row.append(float(np.max(window)))
        if self.add_calendar:
            row.append(t % 7)
            row.append((t % 365) // 30)
            row.append((t % 365) // 90)
        if self.add_fourier:
            row.append(np.sin(2 * np.pi * t / 7))
            row.append(np.cos(2 * np.pi * t / 7))
            row.append(np.sin(2 * np.pi * t / 365.25))
            row.append(np.cos(2 * np.pi * t / 365.25))
            row.append(np.sin(4 * np.pi * t / 365.25))
            row.append(np.cos(4 * np.pi * t / 365.25))
        return np.array([row])

    def get_feature_names(self):
        names = []
        for lag in self.lags:
            names.append(f"lag_{lag}")
        for w in self.rolling_windows:
            for agg in self.rolling_aggs:
                names.append(f"rolling_{w}_{agg}")
        if self.add_calendar:
            names.extend(["dow", "month", "quarter"])
        if self.add_fourier:
            names.extend([
                "sin_7", "cos_7",
                "sin_365", "cos_365",
                "sin2_365", "cos2_365",
            ])
        return names
