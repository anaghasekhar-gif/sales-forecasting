"""
ARIMA / SARIMA Training and Evaluation Module.

This module implements:
1. Stationarity testing using the Augmented Dickey-Fuller (ADF) test.
2. Differencing diagnostics (d=1) to establish stationarity.
3. Fitting a seasonal ARIMA (SARIMA) model on historical training data.
4. Generating out-of-sample test forecasts.
5. Serializing the trained model to models/arima_model.pkl.
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.feature_engineering import chronological_train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_stationarity(series: pd.Series, series_name: str = "Sales Series") -> dict:
    """
    Perform Augmented Dickey-Fuller (ADF) test for stationarity.

    Null Hypothesis (H0): The series has a unit root (is non-stationary).
    Alternative Hypothesis (H1): The series has no unit root (is stationary).
    """
    logger.info(f"Conducting Augmented Dickey-Fuller (ADF) test on {series_name}...")
    clean_series = series.dropna()
    adf_result = adfuller(clean_series, autolag="AIC")

    stat = adf_result[0]
    p_value = adf_result[1]
    lags = adf_result[2]
    n_obs = adf_result[3]
    critical_values = adf_result[4]
    is_stationary = p_value < 0.05

    print(f"\n--- Augmented Dickey-Fuller Test: {series_name} ---")
    print(f"ADF Test Statistic:      {stat:.4f}")
    print(f"p-value:                 {p_value:.4e}")
    print(f"Lags Used:               {lags}")
    print(f"Number of Observations:  {n_obs:,}")
    print("Critical Values:")
    for key, val in critical_values.items():
        print(f"   {key}: {val:.4f}")
    print(f"Stationary at alpha=0.05? {'YES (Reject H0)' if is_stationary else 'NO (Fail to Reject H0)'}\n")

    return {
        "adf_statistic": stat,
        "p_value": p_value,
        "critical_values": critical_values,
        "is_stationary": is_stationary,
    }


def train_arima_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "Sales",
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 0, 1, 7),
    models_dir: str = "models",
) -> tuple[object, np.ndarray, dict]:
    """
    Train a SARIMA model on the training series and forecast for the test horizon.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data.
    test_df : pd.DataFrame
        Testing data (used to determine forecast length).
    target_col : str
        Target sales column.
    order : tuple
        (p, d, q) non-seasonal parameters.
    seasonal_order : tuple
        (P, D, Q, s) seasonal parameters (s=7 for weekly retail cycle).
    models_dir : str
        Output directory to serialize model.

    Returns
    -------
    tuple
        (fitted_model, test_predictions, metadata)
    """
    os.makedirs(models_dir, exist_ok=True)
    y_train = train_df[target_col].values
    forecast_steps = len(test_df)

    logger.info(f"Fitting SARIMAX{order}x{seasonal_order} on {len(y_train)} training observations...")
    model = SARIMAX(
        y_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted_model = model.fit(disp=False)
    logger.info(f"Model converged. AIC: {fitted_model.aic:.2f}, BIC: {fitted_model.bic:.2f}")

    logger.info(f"Generating out-of-sample forecast for {forecast_steps} test steps...")
    forecast_res = fitted_model.get_forecast(steps=forecast_steps)
    test_preds = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int()

    # Save model artifact
    model_path = os.path.join(models_dir, "arima_model.pkl")
    joblib.dump(fitted_model, model_path)
    logger.info(f"ARIMA model successfully serialized to '{model_path}'.")

    # Save out-of-sample test predictions
    pred_df = test_df[["Date", target_col]].copy()
    pred_df["ARIMA_Forecast"] = test_preds
    pred_df["ARIMA_Lower_CI"] = conf_int[:, 0]
    pred_df["ARIMA_Upper_CI"] = conf_int[:, 1]
    pred_path = os.path.join(models_dir, "arima_test_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"Saved ARIMA test predictions to '{pred_path}'.")

    metadata = {
        "order": order,
        "seasonal_order": seasonal_order,
        "aic": float(fitted_model.aic),
        "bic": float(fitted_model.bic),
        "train_size": len(y_train),
        "test_size": forecast_steps,
    }
    return fitted_model, test_preds, metadata


def run_arima_pipeline(data_path: str = "data/sales.csv"):
    """Execute complete ARIMA workflow."""
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])

    # 1. Stationarity Analysis
    raw_sales = df["Sales"]
    check_stationarity(raw_sales, "Raw Historical Sales")

    diff_sales = raw_sales.diff().dropna()
    check_stationarity(diff_sales, "First-Differenced Sales (d=1)")

    # 2. Chronological Split
    train_df, test_df, _ = chronological_train_test_split(df, train_ratio=0.80)

    # 3. Train Model and Generate Test Forecast
    model, preds, meta = train_arima_model(train_df, test_df)

    print("\n--- ARIMA / SARIMA Training Complete ---")
    print(f"AIC: {meta['aic']:.2f}")
    print(f"BIC: {meta['bic']:.2f}")
    print(f"Test Forecast Steps: {meta['test_size']}")
    return model, preds, meta


if __name__ == "__main__":
    run_arima_pipeline()
