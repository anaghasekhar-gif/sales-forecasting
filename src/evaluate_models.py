"""
Model Evaluation and Comparison Module.

This module:
1. Loads actual test observations and out-of-sample forecasts from ARIMA and Prophet.
2. Computes industry-standard error metrics:
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Percentage Error (MAPE)
3. Generates side-by-side model comparison tables.
4. Programmatically selects the best-performing model based on minimal forecasting error.
5. Produces required visualization artifacts:
   - plots/actual_vs_forecast.png
   - plots/model_comparison.png
6. Exports comparison metrics to models/model_comparison_metrics.csv.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", font="sans-serif")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate MAE, RMSE, and MAPE.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth actual values.
    y_pred : np.ndarray
        Model predictions.

    Returns
    -------
    dict
        Computed metrics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Mask zero actuals to prevent division by zero in MAPE calculation
    non_zero_mask = y_true > 0
    if np.sum(non_zero_mask) > 0:
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100.0
    else:
        mape = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
    }


def evaluate_and_compare_models(
    models_dir: str = "models",
    plots_dir: str = "plots",
) -> tuple[pd.DataFrame, str]:
    """
    Load test predictions, evaluate both models, generate plots, and select the champion.
    """
    os.makedirs(plots_dir, exist_ok=True)

    arima_path = os.path.join(models_dir, "arima_test_predictions.csv")
    prophet_path = os.path.join(models_dir, "prophet_test_predictions.csv")

    if not os.path.exists(arima_path) or not os.path.exists(prophet_path):
        raise FileNotFoundError(
            f"Missing prediction files in {models_dir}/. Ensure train_arima.py and train_prophet.py have been executed."
        )

    df_arima = pd.read_csv(arima_path)
    df_prophet = pd.read_csv(prophet_path)

    df_arima["Date"] = pd.to_datetime(df_arima["Date"])
    df_prophet["Date"] = pd.to_datetime(df_prophet["Date"])

    # Align dates
    merged = pd.merge(df_arima, df_prophet[["Date", "Prophet_Forecast", "Prophet_Lower_CI", "Prophet_Upper_CI"]], on="Date")

    y_actual = merged["Sales"].values
    y_arima = merged["ARIMA_Forecast"].values
    y_prophet = merged["Prophet_Forecast"].values

    metrics_arima = calculate_metrics(y_actual, y_arima)
    metrics_prophet = calculate_metrics(y_actual, y_prophet)

    comparison_df = pd.DataFrame([
        {"Model": "ARIMA / SARIMA", "MAE": metrics_arima["MAE"], "RMSE": metrics_arima["RMSE"], "MAPE (%)": metrics_arima["MAPE (%)"]},
        {"Model": "Meta Prophet", "MAE": metrics_prophet["MAE"], "RMSE": metrics_prophet["RMSE"], "MAPE (%)": metrics_prophet["MAPE (%)"]},
    ])

    # Save metrics to CSV
    metrics_path = os.path.join(models_dir, "model_comparison_metrics.csv")
    comparison_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved evaluation metrics to '{metrics_path}'.")

    # Select Best Model
    # We prioritize lower MAPE and lower RMSE
    if metrics_prophet["MAPE (%)"] < metrics_arima["MAPE (%)"]:
        best_model = "Meta Prophet"
        best_mape = metrics_prophet["MAPE (%)"]
    else:
        best_model = "ARIMA / SARIMA"
        best_mape = metrics_arima["MAPE (%)"]

    print("\n" + "=" * 65)
    print("           MODEL PERFORMANCE BENCHMARK COMPARISON")
    print("=" * 65)
    print(comparison_df.to_string(index=False, formatters={
        "MAE": lambda x: f"€{x:,.2f}",
        "RMSE": lambda x: f"€{x:,.2f}",
        "MAPE (%)": lambda x: f"{x:.2f}%"
    }))
    print("=" * 65)
    print(f"Champion Model Selected: {best_model} (Test MAPE: {best_mape:.2f}%)\n")

    # Plot 1: Actual vs Forecasts Overlay
    plot_actual_vs_forecast(merged, plots_dir)

    # Plot 2: Model Comparison Bar Chart
    plot_model_comparison_bars(comparison_df, plots_dir)

    return comparison_df, best_model


def plot_actual_vs_forecast(merged_df: pd.DataFrame, output_dir: str = "plots"):
    """Generate and save actual vs forecasted sales comparison chart."""
    logger.info("Generating plot: actual_vs_forecast.png")
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)

    # Convert values to millions for readable axis
    ax.plot(
        merged_df["Date"],
        merged_df["Sales"] / 1e6,
        color="black",
        linewidth=2.0,
        label="Actual Test Sales",
        zorder=5,
    )
    ax.plot(
        merged_df["Date"],
        merged_df["ARIMA_Forecast"] / 1e6,
        color="#e74c3c",
        linestyle="--",
        linewidth=1.8,
        label="ARIMA / SARIMA Forecast",
        alpha=0.85,
    )
    ax.plot(
        merged_df["Date"],
        merged_df["Prophet_Forecast"] / 1e6,
        color="#2980b9",
        linestyle="-.",
        linewidth=1.8,
        label="Meta Prophet Forecast",
        alpha=0.9,
    )

    # Confidence intervals for Prophet
    ax.fill_between(
        merged_df["Date"],
        merged_df["Prophet_Lower_CI"] / 1e6,
        merged_df["Prophet_Upper_CI"] / 1e6,
        color="#2980b9",
        alpha=0.15,
        label="Prophet 95% Uncertainty Interval",
    )

    ax.set_title("Out-of-Sample Forecasting: Actual vs ARIMA vs Meta Prophet", fontweight="bold", pad=15)
    ax.set_xlabel("Test Period (Date)", labelpad=10)
    ax.set_ylabel("Sales (€ Millions)", labelpad=10)
    ax.legend(frameon=True, facecolor="white", loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)

    save_path = os.path.join(output_dir, "actual_vs_forecast.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_model_comparison_bars(comparison_df: pd.DataFrame, output_dir: str = "plots"):
    """Generate and save multi-metric comparison bar charts."""
    logger.info("Generating plot: model_comparison.png")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    models = comparison_df["Model"].tolist()
    colors = ["#e74c3c", "#2980b9"]

    # MAE Chart
    bars1 = ax1.bar(models, comparison_df["MAE"] / 1e6, color=colors, width=0.5, edgecolor="black")
    ax1.set_title("Mean Absolute Error (MAE)", fontweight="bold")
    ax1.set_ylabel("€ Millions")
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval * 1.02, f"€{yval:.2f}M", ha="center", va="bottom", fontweight="bold")

    # RMSE Chart
    bars2 = ax2.bar(models, comparison_df["RMSE"] / 1e6, color=colors, width=0.5, edgecolor="black")
    ax2.set_title("Root Mean Squared Error (RMSE)", fontweight="bold")
    ax2.set_ylabel("€ Millions")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval * 1.02, f"€{yval:.2f}M", ha="center", va="bottom", fontweight="bold")

    # MAPE Chart
    bars3 = ax3.bar(models, comparison_df["MAPE (%)"], color=colors, width=0.5, edgecolor="black")
    ax3.set_title("Mean Absolute Percentage Error (MAPE)", fontweight="bold")
    ax3.set_ylabel("Percentage (%)")
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, yval * 1.02, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")

    for ax in (ax1, ax2, ax3):
        ax.grid(True, linestyle=":", alpha=0.5)

    save_path = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


if __name__ == "__main__":
    evaluate_and_compare_models()
