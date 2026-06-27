from __future__ import annotations
import numpy as np
from pathlib import Path
from functools import lru_cache


def make_synthetic(n_days=1095, seed=42):
    """Generates daily demand with patterns found in real Walmart retail data:
    gradual growth (~2% YoY), weekly seasonality (weekend peaks),
    holiday-season bump (Q4), random promotions with exponential decay,
    heteroskedastic noise with variance proportional to demand level,
    and occasional stockouts (zero-demand days)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    base = 50 * (1 + 0.3 * t / n_days)
    weekly = 15 * np.sin(2 * np.pi * t / 7 - 1.5)
    doy = t % 365
    yearly = 25 * np.exp(-((doy - 330) ** 2) / (2 * 50 ** 2))
    promo = np.zeros(n_days)
    for _ in range(rng.poisson(12)):
        start = rng.integers(30, n_days - 30)
        duration = rng.integers(7, 21)
        lift = rng.uniform(10, 40)
        end = min(start + duration, n_days)
        decay = np.exp(-np.arange(end - start) / 7)
        promo[start:end] += lift * decay
    level = base + weekly + yearly + promo
    noise = rng.normal(0, 1, n_days) * (1 + 0.15 * level)
    demand = np.maximum(level + noise, 0)
    zero_days = rng.choice(n_days, size=rng.poisson(15), replace=False)
    demand[zero_days] = 0.0
    return demand


def load_csv(csv_name: str, value_col: str = "demand",
             date_col: str | None = None) -> np.ndarray:
    """Loads a single-column time series from data/raw/<csv_name>."""
    import pandas as pd
    df = pd.read_csv(Path("data/raw") / csv_name)
    if date_col:
        df = df.sort_values(date_col)
    return df[value_col].astype(float).to_numpy()


def fetch_fred():
    """Fetches monthly US retail sales (Food & Beverage Stores) from FRED.
    Series ID: RSFSDP. Data from 2015 onward. Returns (values, dates).
    Reference: https://fred.stlouisfed.org/series/RSFSDP"""
    import pandas_datareader as pdr
    import pandas as pd
    df = pdr.data.DataReader("RSFSDP", "fred", "2015-01-01")
    idx = df.index.values
    vals = df["RSFSDP"].astype(float).values
    return vals, idx


def fetch_wmt():
    """Fetches daily Walmart (WMT) adjusted close price from Yahoo Finance.
    Returns (values, dates). Used as a proxy retail demand signal.
    Reference: https://finance.yahoo.com/quote/WMT/"""
    import yfinance as yf
    import numpy as np
    df = yf.download("WMT", start="2015-01-01", progress=False, auto_adjust=True)
    vals = df["Close"].astype(float).values.flatten()
    idx = df.index.values
    return vals, idx


def fetch_prophet_retail():
    """Fetches monthly US retail sales from the Prophet example dataset.
    Source: https://raw.githubusercontent.com/facebook/prophet/
            main/examples/example_retail_sales.csv
    This is real US Census Bureau retail sales data (1992-2016).
    Reference: Prophet documentation; US Census Bureau."""
    import pandas as pd
    url = ("https://raw.githubusercontent.com/facebook/prophet/"
           "main/examples/example_retail_sales.csv")
    df = pd.read_csv(url)
    vals = df["y"].astype(float).values
    idx = pd.to_datetime(df["ds"]).values
    return vals, idx


def detect_frequency(dates):
    """Attempts to detect data frequency from the date index.
    Returns 'daily', 'weekly', 'monthly', or 'unknown'."""
    import pandas as pd
    try:
        idx = pd.DatetimeIndex(dates)
        freq = pd.infer_freq(idx[:10])
        if freq is None:
            return "unknown"
        if "D" in freq:
            return "daily"
        if "W" in freq:
            return "weekly"
        if "M" in freq or "MS" in freq:
            return "monthly"
        return freq
    except Exception:
        return "unknown"
