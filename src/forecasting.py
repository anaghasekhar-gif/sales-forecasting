"""
Future Sales Forecasting Module.

This module:
1. Retrains the champion forecasting model (Meta Prophet or SARIMA) on 100% of historical data.
2. Generates out-of-history future predictions for configurable horizons (7, 30, 60, 90 days).
3. Produces lower and upper confidence/uncertainty intervals.
4. Saves visualization artifact:
   - plots/future_sales_forecast.png
5. Exports future predictions to CSV for business reporting.
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.serialize import model_to_json

from src.feature_engineering import prepare_prophet_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", font="sans-serif")


def retrain_champion_prophet(df: pd.DataFrame, models_dir: str = "models") -> Prophet:
    """
    Retrain Meta Prophet model on the entire historical dataset (all available observations).
    """
    os.makedirs(models_dir, exist_ok=True)
    p_full = prepare_prophet_dataframe(df)

    logger.info(f"Retraining champion Prophet model on 100% of historical data ({len(p_full)} days)...")
    champion = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        interval_width=0.95,
    )
    try:
        champion.add_country_holidays(country_name="DE")
    except Exception:
        pass

    for regressor in ["Promo", "SchoolHoliday"]:
        if regressor in p_full.columns:
            champion.add_regressor(regressor, mode="multiplicative")

    champion.fit(p_full)
    logger.info("Champion model retrained successfully on full history.")

    # Save full retrained model
    json_path = os.path.join(models_dir, "prophet_full_model.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(model_to_json(champion))

    return champion


def generate_future_forecast(
    model: Prophet,
    historical_df: pd.DataFrame,
    horizon_days: int = 30,
    output_dir: str = "plots",
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Generate future forecasts for the specified horizon (e.g. 7, 30, 90 days).

    Parameters
    ----------
    model : Prophet
        Fitted Prophet model.
    historical_df : pd.DataFrame
        Historical sales dataframe.
    horizon_days : int
        Number of future days to project.
    output_dir : str
        Directory to save future forecast plot.
    data_dir : str
        Directory to export CSV forecast.

    Returns
    -------
    pd.DataFrame
        Forecast dataframe with ds, yhat, yhat_lower, yhat_upper.
    """
    os.makedirs(output_dir, exist_ok=True)
    last_date = historical_df["Date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    logger.info(f"Generating {horizon_days}-day future forecast from {future_dates[0].strftime('%Y-%m-%d')} to {future_dates[-1].strftime('%Y-%m-%d')}...")

    future_df = pd.DataFrame({"ds": future_dates})

    # Extrapolate future promotional schedule (retail pattern: active weekdays every other week, Sunday off)
    future_dow = future_df["ds"].dt.dayofweek
    # Standard Rossmann retail promo assumption: Monday-Friday active every 2 weeks, closed Sundays
    week_nums = future_df["ds"].dt.isocalendar().week
    future_df["Promo"] = ((week_nums % 2 == 1) & (future_dow < 5)).astype(int)
    # School holiday indicator (summer holidays for Aug):
    future_df["SchoolHoliday"] = (future_df["ds"].dt.month.isin([7, 8])).astype(int)

    forecast = model.predict(future_df)
    forecast_clean = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_clean = forecast_clean.rename(columns={
        "ds": "Date",
        "yhat": "Forecasted_Sales",
        "yhat_lower": "Lower_CI",
        "yhat_upper": "Upper_CI",
    })
    # Clip negative values
    forecast_clean["Forecasted_Sales"] = forecast_clean["Forecasted_Sales"].clip(lower=0)
    forecast_clean["Lower_CI"] = forecast_clean["Lower_CI"].clip(lower=0)
    forecast_clean["Upper_CI"] = forecast_clean["Upper_CI"].clip(lower=0)

    # Save to CSV
    csv_path = os.path.join(data_dir, f"future_forecast_{horizon_days}d.csv")
    forecast_clean.to_csv(csv_path, index=False)
    logger.info(f"Exported future forecast to '{csv_path}'.")

    # Plot future sales forecast
    plot_future_forecast(historical_df, forecast_clean, horizon_days, output_dir)

    return forecast_clean


def plot_future_forecast(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    horizon_days: int,
    output_dir: str = "plots",
):
    """Plot historical sales context with future forecast and uncertainty envelope."""
    logger.info("Generating plot: future_sales_forecast.png")
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)

    # Show last 90 days of history for immediate visual context
    recent_history = historical_df.iloc[-90:].copy()

    ax.plot(
        recent_history["Date"],
        recent_history["Sales"] / 1e6,
        color="#2c3e50",
        linewidth=1.8,
        label="Recent Actual Sales (Last 90 Days)",
    )
    ax.plot(
        forecast_df["Date"],
        forecast_df["Forecasted_Sales"] / 1e6,
        color="#27ae60",
        linestyle="--",
        linewidth=2.2,
        marker="o",
        markersize=4,
        label=f"Projected Future Sales ({horizon_days}-Day Horizon)",
    )

    # Uncertainty envelope
    ax.fill_between(
        forecast_df["Date"],
        forecast_df["Lower_CI"] / 1e6,
        forecast_df["Upper_CI"] / 1e6,
        color="#27ae60",
        alpha=0.18,
        label="95% Predictive Uncertainty Interval",
    )

    # Divider line
    divider = recent_history["Date"].max()
    ax.axvline(divider, color="#e74c3c", linestyle=":", linewidth=1.5, label="Forecast Horizon Boundary")

    ax.set_title(f"Future Sales Forecast: Next {horizon_days} Days Out-of-Sample Projection", fontweight="bold", pad=15)
    ax.set_xlabel("Date", labelpad=10)
    ax.set_ylabel("Sales (€ Millions)", labelpad=10)
    ax.legend(frameon=True, facecolor="white", loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)

    save_path = os.path.join(output_dir, "future_sales_forecast.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def run_future_forecasting(horizon_days: int = 30):
    """Execute complete future forecasting pipeline."""
    df = pd.read_csv("data/sales.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    champion_model = retrain_champion_prophet(df)
    forecast_df = generate_future_forecast(champion_model, df, horizon_days=horizon_days)

    print("\n" + "=" * 65)
    print(f"      FUTURE SALES PROJECTIONS: NEXT {horizon_days} DAYS")
    print("=" * 65)
    print(f"Horizon:                   {forecast_df['Date'].min().strftime('%Y-%m-%d')} to {forecast_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Projected Total Revenue:   €{forecast_df['Forecasted_Sales'].sum():,.2f}")
    print(f"Average Daily Sales:       €{forecast_df['Forecasted_Sales'].mean():,.2f}")
    print(f"Minimum Projected Day:     €{forecast_df['Forecasted_Sales'].min():,.2f}")
    print(f"Maximum Projected Day:     €{forecast_df['Forecasted_Sales'].max():,.2f}")
    print("\nFirst 7 Projected Days:")
    print(forecast_df.head(7).to_string(index=False, formatters={
        "Date": lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"),
        "Forecasted_Sales": lambda x: f"€{x:,.2f}",
        "Lower_CI": lambda x: f"€{x:,.2f}",
        "Upper_CI": lambda x: f"€{x:,.2f}",
    }))
    return forecast_df


if __name__ == "__main__":
    run_future_forecasting(horizon_days=30)
