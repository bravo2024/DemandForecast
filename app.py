from __future__ import annotations

import sys, os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

from src.data import make_synthetic, fetch_fred, fetch_wmt, fetch_prophet_retail
from src.core import rmse, mae, smape, mase, acf, pacf
from src.diagnostics import ljung_box, adf_test
from src.models import (
    SeasonalNaive,
    RidgeForecaster,
    RandomForestRegressor,
    HoltWinters,
    ARIMAForecaster,
    SARIMAForecaster,
    compare_models,
)
from src.validation import walk_forward, evaluate_at_horizons, prediction_intervals

st.set_page_config(page_title="DemandForecast — Walmart", layout="wide")
st.title("DemandForecast — Walmart")
st.markdown(
    "_Retail demand forecasting with walk-forward validation, "
    "multiple model families, and prediction intervals._"
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def unit_label(source):
    if "FRED" in source or "Prophet" in source:
        return "months"
    return "days"

def period_for(source):
    if "WMT" in source:
        return 5
    if "FRED" in source or "Prophet" in source:
        return 12
    return 7

# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Data Source")
    data_source = st.selectbox(
        "Choose a dataset",
        ["Synthetic Demand (demo)",
         "FRED Retail Sales (live)",
         "WMT Stock Price (live)",
         "Prophet Retail Sales (live)"],
        help="Synthetic: zero-download demo.  FRED: real US retail sales "
             "(monthly).  WMT: daily Walmart stock price.  "
             "Prophet: US Census retail data (monthly).",
    )

    per = period_for(data_source)

    n_days = st.slider("Series length", 100, 2000, 800, step=50,
                       help="Only applies to synthetic data; live data uses its full range.")

    st.header("Forecast Settings")
    horizon = st.slider(f"Forecast horizon ({unit_label(data_source)})",
                        1, 60, 14)
    n_windows = st.slider("Walk-forward folds", 2, 10, 5)

    st.header("Models")
    use_sn = st.checkbox("Seasonal Naive", True,
                         help="Baseline forecast equal to the last same-weekday value.")
    use_ridge = st.checkbox("Ridge (linear baseline)", True)
    use_rf = st.checkbox("Random Forest", False,
                         help="Non-linear ensemble.  Slower but captures feature interactions.")
    use_hw = st.checkbox("Holt-Winters", True,
                         help="Exponential smoothing with level, trend, and seasonal components.")
    use_arima = st.checkbox("ARIMA", False,
                            help="Autoregressive Integrated Moving Average.  "
                                 "Uses statsmodels for estimation.")
    use_sarima = st.checkbox("SARIMA", False,
                             help="Seasonal ARIMA.  Captures weekly or yearly periodicity.")

    arima_p = st.slider("ARIMA p (AR order)", 0, 5, 1, disabled=not use_arima)
    arima_d = st.slider("ARIMA d (differencing)", 0, 2, 1, disabled=not use_arima)
    arima_q = st.slider("ARIMA q (MA order)", 0, 5, 1, disabled=not use_arima)

    sarima_s = st.slider("SARIMA s (seasonal period)", 4, 52,
                         value=per, disabled=not use_sarima,
                         help="The seasonal period is auto-set from the "
                              "data source (7=daily, 5=WMT trading, "
                              "12=monthly). Adjust manually if needed.")

    run = st.button("Run Full Pipeline", type="primary")

# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Fetching live data...")
def _fetch(source):
    if source == "FRED Retail Sales (live)":
        return fetch_fred()
    if source == "WMT Stock Price (live)":
        return fetch_wmt()
    if source == "Prophet Retail Sales (live)":
        return fetch_prophet_retail()
    raise ValueError("unknown source")

if "FRED" in data_source or "WMT" in data_source or "Prophet" in data_source:
    try:
        series, dates = _fetch(data_source)
        st.sidebar.success(f"Loaded {len(series)} points")
    except Exception as e:
        st.sidebar.error(f"Live data failed: {e}")
        st.sidebar.info("Falling back to synthetic data.")
        series = make_synthetic(n_days, 42)
        dates = pd.date_range("2015-01-01", periods=len(series), freq="D")
else:
    series = make_synthetic(n_days, 42)
    dates = pd.date_range("2015-01-01", periods=len(series), freq="D")

freq_str = unit_label(data_source)
unit_lbl = unit_label(data_source)

# ---------------------------------------------------------------------------
# tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Data Explorer",
    "Seasonality & ACF/PACF",
    "Model Training",
    "Forecast & Uncertainty",
])

# ===== Tab 1 — Data Explorer =====

with tab1:
    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        st.subheader("Demand / Sales Series")
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(series, lw=0.7, color="#1f77b4")
        ma7 = np.convolve(series, np.ones(per) / per, mode="valid")
        offset = per - 1
        ax.plot(np.arange(offset, offset + len(ma7)), ma7, lw=1.6,
                color="#d62728", label=f"{per}-period MA")
        ax.set_xlabel(f"Time ({unit_lbl})")
        ax.set_ylabel("Value")
        ax.legend(loc="upper left", framealpha=0.9)
        st.pyplot(fig)

    with col2:
        st.subheader("Summary Statistics")
        nz = series[series > 0]
        st.metric("Mean", f"{np.mean(series):.2f}")
        st.metric("Std Dev", f"{np.std(series):.2f}")
        st.metric(f"Zero values", f"{(series == 0).sum()} ({(series == 0).mean() * 100:.1f}%)")
        st.metric("Max", f"{series.max():.2f}")
        st.metric("Min (non-zero)", f"{nz.min():.2f}" if len(nz) else "N/A")

    st.markdown("---")
    st.subheader("Stationarity Test (Augmented Dickey-Fuller)")

    st.latex(
        r"\Delta y_t = \alpha + \beta t + \gamma y_{t-1}"
        r"+ \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \varepsilon_t"
    )
    st.markdown(
        "Null hypothesis $H_0$: the series has a unit root "
        "(non-stationary, $\\gamma = 0$).  "
        "A test statistic more negative than the 5\\% critical value "
        "rejects $H_0$, indicating stationarity."
    )

    adf = adf_test(series)
    if adf["t_stat"] is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ADF Statistic", f"{adf['t_stat']:.3f}")
        c2.metric("p-value", "< 0.05" if adf["is_stationary"] else "> 0.05")
        c3.metric("Lag selected", str(adf["lag"]))
        c4.metric("Verdict",
                  "Stationary" if adf["is_stationary"] else "Non-stationary",
                  help="If non-stationary, differencing (d >= 1) is recommended "
                       "before fitting ARIMA models.")
        cv_vals = adf["critical_values"]
        st.caption(
            f"Critical values:  1% = {cv_vals.get('1', '—')},  "
            f"5% = {cv_vals.get('5', '—')},  10% = {cv_vals.get('10', '—')}"
        )
    else:
        st.warning("ADF test could not be computed on this series.")

    with st.expander("Practitioner Note"):
        st.markdown(
            "In retail demand forecasting, stationarity is rarely assumed.  "
            "Raw sales data typically exhibits trend and seasonality.  "
            "The ADF test informs how many differencing passes are needed "
            "before fitting ARIMA models.  For strongly seasonal series, "
            "a seasonal differencing pass may be more appropriate than "
            "a first-order (lag-1) difference."
        )

# ===== Tab 2 — Seasonality & ACF/PACF =====

with tab2:
    st.subheader("Seasonal Decomposition")
    try:
        decomp = seasonal_decompose(series, model="additive", period=per)
        fig2, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
        titles = ["Observed", "Trend", "Seasonal", "Residual"]
        for ax_i, comp, title in zip(axes,
                                     [decomp.observed, decomp.trend,
                                      decomp.seasonal, decomp.resid],
                                     titles):
            ax_i.plot(comp, lw=0.7)
            ax_i.set_ylabel(title)
        axes[0].set_title("Seasonal Decomposition (additive model)")
        plt.tight_layout()
        st.pyplot(fig2)
    except Exception as exc:
        st.info(f"Decomposition not available: {exc}")

    st.markdown("---")
    st.subheader("Autocorrelation Function (ACF)")

    st.latex(
        r"\rho_k = \frac{\text{Cov}(y_t, y_{t-k})}{\text{Var}(y_t)}"
        r" = \frac{\sum_{t=k+1}^n (y_t - \bar{y})(y_{t-k} - \bar{y})}"
        r"{\sum_{t=1}^n (y_t - \bar{y})^2}"
    )
    st.markdown(
        "The ACF measures the correlation between the series and its lagged "
        "values.  A slow decay suggests non-stationarity; a sharp cutoff at "
        "lag $q$ hints at an MA($q$) process."
    )

    nlags = min(40, len(series) // 4)
    acf_vals = acf(series, nlags)
    conf = 1.96 / np.sqrt(len(series))

    fig3, ax3 = plt.subplots(figsize=(10, 3))
    lags = np.arange(nlags + 1)
    ax3.vlines(lags, 0, acf_vals, colors="#1f77b4", lw=1.2)
    ax3.plot(lags, acf_vals, "o", color="#1f77b4", ms=4)
    ax3.axhline(0, color="gray", lw=0.5)
    ax3.axhline(conf, color="red", ls="--", lw=0.7, label="95% confidence")
    ax3.axhline(-conf, color="red", ls="--", lw=0.7)
    ax3.set_xlabel("Lag")
    ax3.set_ylabel("ACF")
    ax3.legend(framealpha=0.9)
    st.pyplot(fig3)

    st.subheader("Partial Autocorrelation Function (PACF)")

    st.latex(
        r"\phi_{kk} = \text{Corr}(y_t, y_{t-k} \mid y_{t-1}, \dots, y_{t-k+1})"
    )
    st.markdown(
        "The PACF measures the correlation between $y_t$ and $y_{t-k}$ "
        "after removing the effect of intermediate lags.  A sharp cutoff at "
        "lag $p$ suggests an AR($p$) process."
    )

    pacf_vals = pacf(series, nlags)

    fig4, ax4 = plt.subplots(figsize=(10, 3))
    ax4.vlines(lags, 0, pacf_vals, colors="#2ca02c", lw=1.2)
    ax4.plot(lags, pacf_vals, "o", color="#2ca02c", ms=4)
    ax4.axhline(0, color="gray", lw=0.5)
    ax4.axhline(conf, color="red", ls="--", lw=0.7, label="95% confidence")
    ax4.axhline(-conf, color="red", ls="--", lw=0.7)
    ax4.set_xlabel("Lag")
    ax4.set_ylabel("PACF")
    ax4.legend(framealpha=0.9)
    st.pyplot(fig4)

    with st.expander("Interpreting ACF / PACF"):
        st.markdown(
            "**Box-Jenkins methodology:**  "
            "1) A slowly decaying ACF suggests the series needs differencing.  "
            "2) A PACF that cuts off at lag $p$ suggests an AR($p$) model.  "
            "3) An ACF that cuts off at lag $q$ suggests an MA($q$) model.  "
            "4) Seasonal patterns appear at lags 7, 14, 28, etc., suggesting "
            "a SARIMA model with $s=7$.\n\n"
            "Reference: Box, Jenkins, Reinsel 'Time Series Analysis' (4th ed.), Ch. 6."
        )

# ===== Tab 3 — Model Training =====

with tab3:
    st.subheader("Walk-Forward Validation")

    selected = {}
    if use_sn:
        selected["SeasonalNaive"] = SeasonalNaive(per)
    if use_ridge:
        selected["Ridge"] = RidgeForecaster()
    if use_rf:
        selected["RF"] = RandomForestRegressor(n_trees=20)
    if use_hw:
        selected["HoltWinters"] = HoltWinters(per)
    if use_arima:
        selected["ARIMA"] = ARIMAForecaster(order=(arima_p, arima_d, arima_q))
    if use_sarima:
        sel_order = (arima_p, arima_d, arima_q)
        sel_seas = (1, 0, 1, sarima_s)
        selected["SARIMA"] = SARIMAForecaster(order=sel_order,
                                               seasonal_order=sel_seas)

    if not selected:
        st.warning("At least one model must be selected.")
    elif not run:
        st.info("Models are configured in the sidebar.  "
                "Click **Run Full Pipeline** to begin training.")
    else:
        with st.spinner("Training and evaluating across folds..."):
            results = compare_models(series, horizon, n_windows, selected)

        rows = []
        best_rmse = float("inf")
        best_name = None
        for name, r in results.items():
            if np.isfinite(r["rmse"]):
                rows.append({
                    "Model": name,
                    "RMSE": f"{r['rmse']:.2f}",
                    "MAE": f"{r['mae']:.2f}",
                    "sMAPE": f"{r['smape']:.1f}%",
                    "Status": "OK",
                })
                if r["rmse"] < best_rmse:
                    best_rmse = r["rmse"]
                    best_name = name
            else:
                rows.append({"Model": name, "RMSE": "\u2014", "MAE": "\u2014",
                             "sMAPE": "\u2014", "Status": "FAILED"})

        st.table(rows)

        if best_name:
            st.success(f"**Best model by RMSE**: {best_name}  "
                       f"(RMSE = {best_rmse:.2f})")

            fig5, ax5 = plt.subplots(figsize=(8, 3.5))
            ok = [r for r in rows if r["Status"] == "OK"]
            names = [r["Model"] for r in ok]
            vals = [float(r["RMSE"]) for r in ok]
            colors = ["#2ca02c" if n == best_name else "#1f77b4" for n in names]
            ax5.barh(names, vals, color=colors)
            ax5.set_xlabel("RMSE (lower is better)")
            ax5.invert_yaxis()
            st.pyplot(fig5)

            best_mod = selected[best_name]
            h_err = evaluate_at_horizons(series, best_mod, horizon, n_windows)
            valid_h = [(h, m) for h, m in zip(h_err["horizon"], h_err["mae_by_horizon"])
                       if np.isfinite(m)]
            if valid_h:
                fig6, ax6 = plt.subplots(figsize=(8, 3))
                ax6.plot([v[0] for v in valid_h], [v[1] for v in valid_h],
                         marker="o", lw=1.5)
                ax6.set_xlabel(f"Forecast horizon ({unit_lbl} ahead)")
                ax6.set_ylabel("MAE")
                ax6.grid(alpha=0.3)
                ax6.set_title("Forecast Accuracy by Horizon")
                st.pyplot(fig6)

        with st.expander("Model equations"):
            if use_sn:
                st.markdown("**Seasonal Naive**")
                st.latex(r"\hat{y}_{t+h} = y_{t+h - m}")
                st.markdown("where $m$ is the seasonal period (e.g., 7 for weekly).")
            if use_hw:
                st.markdown("**Holt-Winters (additive)**")
                st.latex(r"\ell_t = \alpha (y_t - s_{t-m})"
                         r"+ (1-\alpha)(\ell_{t-1} + b_{t-1})")
                st.latex(r"b_t = \beta (\ell_t - \ell_{t-1})"
                         r"+ (1-\beta) b_{t-1}")
                st.latex(r"s_t = \gamma (y_t - \ell_t)"
                         r"+ (1-\gamma) s_{t-m}")
                st.markdown("Forecast:  "
                            "$\\hat{y}_{t+h|t} = \\ell_t + h \\cdot b_t + s_{t-m+h}$")
                st.caption("Reference: Hyndman & Athanasopoulos Ch. 8.")
            if use_arima:
                st.markdown("**ARIMA(p,d,q)**")
                st.latex(
                    r"(1 - \phi_1 B - \cdots - \phi_p B^p)"
                    r"(1-B)^d y_t = c"
                    r"+ (1 + \theta_1 B + \cdots + \theta_q B^q) \varepsilon_t"
                )
                st.markdown("where $B$ is the backshift operator, "
                            "$By_t = y_{t-1}$.")
                st.caption("Reference: Box-Jenkins Ch. 6-7; Hyndman & Athanasopoulos Ch. 9.")
            if use_sarima:
                st.markdown("**SARIMA(p,d,q)(P,D,Q)\\_s**")
                st.latex(
                    r"\Phi(B) \Phi_s(B^s) (1-B)^d (1-B^s)^D y_t"
                    r"= \Theta(B) \Theta_s(B^s) \varepsilon_t"
                )
                st.markdown("Adds seasonal AR, differencing, and MA terms "
                            "to capture periodic patterns.")
                st.caption("Reference: Box-Jenkins Ch. 9.")
            if use_ridge:
                st.markdown("**Ridge Regression**")
                st.latex(r"\hat{y} = X \hat{\beta}, \quad"
                         r"\hat{\beta} = (X^\top X + \alpha I)^{-1} X^\top y")
                st.markdown("Features include lags, rolling window statistics, "
                            "calendar dummies, and Fourier terms.")
            if use_rf:
                st.markdown("**Random Forest**")
                st.markdown(
                    "An ensemble of $n$ regression trees.  Each tree is grown "
                    "on a bootstrap sample; node splits minimize within-node "
                    "variance.  The forest prediction is the mean across trees.  "
                    "Reference: Breiman (2001) 'Random Forests'."
                )

        with st.expander("Per-fold error trajectory"):
            fig7, ax7 = plt.subplots(figsize=(9, 3.5))
            for name, r in results.items():
                if not np.isfinite(r["rmse"]):
                    continue
                folds = [e["fold"] for e in r["errors"]
                         if np.isfinite(e.get("rmse", np.nan))]
                vals = [e["rmse"] for e in r["errors"]
                        if np.isfinite(e.get("rmse", np.nan))]
                ax7.plot(folds, vals, marker=".", label=name)
            ax7.set_xlabel("Fold"); ax7.set_ylabel("RMSE")
            ax7.legend(); ax7.grid(alpha=0.3)
            st.pyplot(fig7)

# ===== Tab 4 — Forecast & Uncertainty =====

with tab4:
    st.subheader(f"Demand Forecast — Next {horizon} {unit_lbl}")

    if not run:
        st.info("Click **Run Full Pipeline** to generate forecasts.")
    elif not best_name:
        st.error("No model succeeded.  Check model settings and try again.")
    else:
        best_mod_full = selected[best_name]
        best_mod_full.fit(series, horizon)
        forecast = best_mod_full.predict(horizon)
        intervals = prediction_intervals(series, best_mod_full, horizon,
                                         n_simulations=200)

        plot_intervals = intervals
        if intervals and st.checkbox("Apply GARCH volatility adjustment",
                                     value=False,
                                     help="Uses a GARCH(1,1) model on residuals "
                                          "to widen/narrow prediction intervals "
                                          "based on recent forecast volatility."):
            wf = walk_forward(best_mod_full, series, horizon, n_windows)
            residuals = np.array([
                a - p for e in wf
                if "pred" in e and "actual" in e
                for a, p in zip(e["actual"], e["pred"])
            ])
            if len(residuals) > 20:
                from src.models import GARCHResiduals
                garch = GARCHResiduals()
                garch.fit(residuals)
                garch_var = garch.forecast_variance(horizon)
                for i, p in enumerate(intervals):
                    spread = (p["upper_90"] - p["lower_90"])
                    garch_spread = spread * np.sqrt(garch_var[i] /
                                                     (np.mean(garch_var) + 1e-10))
                    midpoint = (p["upper_90"] + p["lower_90"]) / 2
                    p["lower_90"] = max(midpoint - garch_spread / 2, 0.0)
                    p["upper_90"] = midpoint + garch_spread / 2
                plot_intervals = intervals

        fig8, ax8 = plt.subplots(figsize=(12, 4.5))
        lookback = min(180, len(series))
        hist = series[-lookback:]
        ax8.plot(np.arange(-lookback, 0), hist, lw=0.8, color="#1f77b4",
                 label="History")
        ax8.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.6)

        fwd = np.arange(1, horizon + 1)
        ax8.plot(fwd, forecast, lw=1.8, color="#d62728",
                 label=f"Forecast ({best_name})")

        if plot_intervals:
            lower = np.array([p["lower_90"] for p in plot_intervals])
            upper = np.array([p["upper_90"] for p in plot_intervals])
            ax8.fill_between(fwd, lower, upper, alpha=0.2, color="#d62728",
                             label="90% PI")

        ax8.set_xlabel(f"Time from now ({unit_lbl})")
        ax8.set_ylabel("Demand / Sales")
        ax8.legend(loc="upper left", framealpha=0.9)
        ax8.grid(alpha=0.25)
        st.pyplot(fig8)

        st.subheader("Point Forecast & Prediction Intervals")
        if plot_intervals:
            tbl = []
            for p in plot_intervals:
                tbl.append({
                    "t+" if unit_lbl == "days" else "t+": str(p["horizon"]),
                    "Forecast": f"{p['forecast']:.2f}",
                    "Lower 90%": f"{p['lower_90']:.2f}",
                    "Upper 90%": f"{p['upper_90']:.2f}",
                })
            st.table(tbl)

            csv_lines = ["horizon,forecast,lower_90,upper_90"]
            for p in plot_intervals:
                csv_lines.append(
                    f"{p['horizon']},{p['forecast']:.4f},"
                    f"{p['lower_90']:.4f},{p['upper_90']:.4f}"
                )
            csv_str = "\n".join(csv_lines)
            st.download_button("Download forecast as CSV", data=csv_str,
                               file_name="demand_forecast.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("Uncertainty Horizon Decay")
        h_err = evaluate_at_horizons(series, best_mod_full, horizon, n_windows)
        valid_h = [(h, m) for h, m in zip(h_err["horizon"], h_err["mae_by_horizon"])
                   if np.isfinite(m)]
        if valid_h:
            fig9, ax9 = plt.subplots(figsize=(8, 3))
            ax9.plot([v[0] for v in valid_h], [v[1] for v in valid_h],
                     marker="o", color="#d62728", lw=1.5)
            ax9.set_xlabel(f"Horizon ({unit_lbl} ahead)")
            ax9.set_ylabel("MAE")
            ax9.grid(alpha=0.3)
            ax9.set_title("Forecast Accuracy Decays with Horizon")
            st.pyplot(fig9)

            with st.expander("Implications for demand planning"):
                st.markdown(
                    "Prediction intervals widening at longer horizons is a "
                    "fundamental property of forecasting.  For retail "
                    "inventory decisions, this means safety stock levels "
                    "must increase non-linearly with lead time.  A forecast "
                    "with $\\pm$10 units at t+1 may widen to $\\pm$30 at "
                    "t+14, meaning the inventory commitment carries "
                    "substantially more risk for long-leadtime orders."
                )

        with st.expander("Residual diagnostics (best model)"):
            wf_errors = walk_forward(best_mod_full, series, horizon, n_windows)
            residuals = []
            for e in wf_errors:
                if "pred" in e and "actual" in e:
                    for a, p in zip(e["actual"], e["pred"]):
                        residuals.append(a - p)
            residuals = np.array(residuals)

            if len(residuals) > 5:
                q_stat, p_val = ljung_box(residuals, nlags=min(20, len(residuals) // 5))
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Ljung-Box Q-stat",
                              f"{q_stat:.2f}",
                              help="Tests whether residuals are white noise.  "
                                   "A p-value < 0.05 suggests leftover autocorrelation.")
                with col_b:
                    p_display = f"{p_val:.4f}" if p_val is not None else "N/A"
                    st.metric("p-value", p_display)

                fig10, axes10 = plt.subplots(1, 2, figsize=(10, 3.5))
                axes10[0].hist(residuals, bins=25, edgecolor="white", alpha=0.7)
                axes10[0].axvline(0, color="red", ls="--", lw=0.8)
                axes10[0].set_xlabel("Residual (actual \u2212 forecast)")
                axes10[0].set_ylabel("Frequency")
                axes10[1].scatter(range(len(residuals)), residuals, s=4, alpha=0.5)
                axes10[1].axhline(0, color="red", ls="--", lw=0.8)
                axes10[1].set_xlabel("Observation"); axes10[1].set_ylabel("Residual")
                plt.tight_layout()
                st.pyplot(fig10)

                with st.expander("Residual interpretation"):
                    st.markdown(
                        "Residuals should be roughly symmetric around zero "
                        "(mean $\\approx 0$).  A Ljung-Box p-value above 0.05 "
                        "suggests the model has captured all temporal structure.  "
                        "A funnel-shaped scatter plot indicates heteroskedasticity "
                        "(variance changes with level), which is common in "
                        "retail demand data and may be addressed with log "
                        "transformations or GARCH modeling."
                    )
