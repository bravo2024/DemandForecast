from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from src.data import make_synthetic
from src.models import (
    SeasonalNaive,
    RidgeForecaster,
    RandomForestRegressor,
    HoltWinters,
    ARIMAForecaster,
    SARIMAForecaster,
    CausalTransformerForecaster,
    compare_models,
)
from src.persist import save_model
from src.core import rmse
from src.evaluate import save_metrics, print_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train demand forecasting models via walk-forward validation."
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["naive", "ridge", "hw"],
        choices=["naive", "ridge", "rf", "hw", "arima", "sarima", "transformer"],
        help="Model(s) to train.  Multiple values accepted.  "
             "Options: naive, ridge, rf, hw, arima, sarima, transformer.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=14,
        help="Forecast horizon in time steps (default: 14).",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=5,
        help="Number of walk-forward validation windows (default: 5).",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=800,
        help="Length of the synthetic demand series (default: 800).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible synthetic data (default: 42).",
    )
    parser.add_argument(
        "--arima-p",
        type=int,
        default=1,
        help="AR order for ARIMA / SARIMA (default: 1).",
    )
    parser.add_argument(
        "--arima-d",
        type=int,
        default=1,
        help="Differencing order for ARIMA / SARIMA (default: 1).",
    )
    parser.add_argument(
        "--arima-q",
        type=int,
        default=1,
        help="MA order for ARIMA / SARIMA (default: 1).",
    )
    parser.add_argument(
        "--sarima-s",
        type=int,
        default=7,
        help="Seasonal period for SARIMA (default: 7).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "models"),
        help="Directory where trained model files are saved "
             "(default: <project>/models/).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    min_train = max(args.horizon * 3, 100)
    if args.length < min_train + args.horizon * args.windows:
        sys.exit(
            f"Series length ({args.length}) too short for "
            f"{args.windows} windows of horizon {args.horizon}.  "
            f"Need at least {min_train + args.horizon * args.windows}.  "
            "Increase --length or decrease --windows / --horizon."
        )

    print(f"Generating synthetic demand series: n={args.length}, seed={args.seed}")
    series = make_synthetic(args.length, args.seed)
    print(f"  mean={np.mean(series):.2f}, std={np.std(series):.2f}  "
          f"(non-zero: {(series > 0).sum()})")

    models: dict[str, object] = {}
    for m in args.models:
        if m == "naive":
            models["SeasonalNaive"] = SeasonalNaive(seasonality=7)
        elif m == "ridge":
            models["Ridge"] = RidgeForecaster()
        elif m == "rf":
            models["RF"] = RandomForestRegressor(n_trees=20)
        elif m == "hw":
            models["HoltWinters"] = HoltWinters(seasonality=7)
        elif m == "arima":
            models["ARIMA"] = ARIMAForecaster(
                order=(args.arima_p, args.arima_d, args.arima_q)
            )
        elif m == "sarima":
            models["SARIMA"] = SARIMAForecaster(
                order=(args.arima_p, args.arima_d, args.arima_q),
                seasonal_order=(1, 0, 1, args.sarima_s),
            )
        elif m == "transformer":
            models["CausalTransformer"] = CausalTransformerForecaster(
                window=28, hidden=32, n_heads=2
            )

    print(f"\nRunning walk-forward validation: horizon={args.horizon}, "
          f"windows={args.windows}")
    print(f"Models: {', '.join(models.keys())}")

    results = compare_models(series, args.horizon, args.windows, models)

    print("\n" + "=" * 60)
    print(f"{'Model':<20} {'RMSE':>10} {'MAE':>10} {'sMAPE':>8}")
    print("=" * 60)

    best_name = None
    best_rmse = float("inf")
    summary_metrics: dict = {"horizon": int(args.horizon), "windows": int(args.windows)}
    for name, r in results.items():
        metric_rmse = r.get("rmse", np.nan)
        metric_mae = r.get("mae", np.nan)
        metric_smape = r.get("smape", np.nan)

        rmse_str = f"{metric_rmse:.2f}" if np.isfinite(metric_rmse) else "——"
        mae_str = f"{metric_mae:.2f}" if np.isfinite(metric_mae) else "——"
        smape_str = f"{metric_smape:.1f}%" if np.isfinite(metric_smape) else "——"

        print(f"{name:<20} {rmse_str:>10} {mae_str:>10} {smape_str:>8}")

        if np.isfinite(metric_rmse):
            summary_metrics[name] = {
                "rmse": float(metric_rmse),
                "mae": float(metric_mae) if np.isfinite(metric_mae) else None,
                "smape": float(metric_smape) if np.isfinite(metric_smape) else None,
            }
            if metric_rmse < best_rmse:
                best_rmse = metric_rmse
                best_name = name

    print("=" * 60)
    if best_name:
        print(f"Best model: {best_name} (RMSE = {best_rmse:.2f})")
        summary_metrics["best_model"] = best_name
    else:
        print("All models failed — check data or hyperparameters.")
    save_metrics(summary_metrics)
    print(f"\nMetrics saved to {os.path.join(args.output_dir, 'metrics.json')}")

    os.makedirs(args.output_dir, exist_ok=True)

    for name, model in models.items():
        if name in results and np.isfinite(results[name].get("rmse", np.nan)):
            path = os.path.join(args.output_dir, f"{name.lower()}.pkl")
            save_model(model, path)
            rmse_val = results[name]["rmse"]
            print(f"  Saved {name} to {path}  (RMSE = {rmse_val:.2f})")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
