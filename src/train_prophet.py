"""
Meta Prophet Training and Evaluation Module.

This module implements:
1. Preparation of time series into Prophet schema: 'ds' (Date), 'y' (Sales).
2. Addition of weekly and yearly seasonalities.
3. Addition of German nationwide holiday effects.
4. Incorporation of promotional and school holiday indicators as exogenous regressors.
5. Multiplicative seasonality modeling to capture proportional promotional surges.
6. Training on historical 80% split and forecasting the 20% test horizon.
7. Serialization to both JSON (prophet native) and PKL (joblib).
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json

from src.feature_engineering import chronological_train_test_split, prepare_prophet_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_prophet_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    models_dir: str = "models",
) -> tuple[Prophet, pd.DataFrame, dict]:
    """
    Configure, fit, and forecast with Meta Prophet.
    """
    os.makedirs(models_dir, exist_ok=True)

    # Prepare training and testing Prophet dataframes
    p_train = prepare_prophet_dataframe(train_df)
    p_test = prepare_prophet_dataframe(test_df)

    logger.info("Initializing Prophet model with yearly, weekly seasonality and multiplicative mode...")
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        interval_width=0.95,
    )

    # Add German country holidays
    try:
        model.add_country_holidays(country_name="DE")
        logger.info("Added German federal holidays to Prophet model.")
    except Exception as e:
        logger.warning(f"Could not load country holidays: {e}")

    # Add exogenous regressors
    for regressor in ["Promo", "SchoolHoliday"]:
        if regressor in p_train.columns:
            model.add_regressor(regressor, mode="multiplicative")
            logger.info(f"Added '{regressor}' as an exogenous regressor.")

    logger.info(f"Fitting Prophet model on {len(p_train)} training records...")
    model.fit(p_train)
    logger.info("Prophet model successfully fitted.")

    # Predict test period
    logger.info(f"Forecasting out-of-sample test horizon ({len(p_test)} days)...")
    forecast = model.predict(p_test)

    # Save model artifacts (both JSON and PKL)
    json_path = os.path.join(models_dir, "prophet_model.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(model_to_json(model))
    logger.info(f"Prophet model serialized to JSON at '{json_path}'.")

    pkl_path = os.path.join(models_dir, "prophet_model.pkl")
    try:
        joblib.dump(model, pkl_path)
        logger.info(f"Prophet model serialized to PKL at '{pkl_path}'.")
    except Exception as e:
        logger.warning(f"Could not pickle Prophet model directly: {e}")

    # Save test predictions
    test_preds_df = test_df[["Date", "Sales"]].copy()
    test_preds_df["Prophet_Forecast"] = forecast["yhat"].values
    test_preds_df["Prophet_Lower_CI"] = forecast["yhat_lower"].values
    test_preds_df["Prophet_Upper_CI"] = forecast["yhat_upper"].values

    pred_path = os.path.join(models_dir, "prophet_test_predictions.csv")
    test_preds_df.to_csv(pred_path, index=False)
    logger.info(f"Saved Prophet test predictions to '{pred_path}'.")

    metadata = {
        "model_type": "Prophet (Multiplicative)",
        "train_rows": len(p_train),
        "test_rows": len(p_test),
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "regressors": ["Promo", "SchoolHoliday"],
    }
    return model, forecast, metadata


def load_prophet_model(models_dir: str = "models") -> Prophet:
    """Load the trained Prophet model from JSON or PKL."""
    json_path = os.path.join(models_dir, "prophet_model.json")
    pkl_path = os.path.join(models_dir, "prophet_model.pkl")

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return model_from_json(f.read())
    elif os.path.exists(pkl_path):
        return joblib.load(pkl_path)
    else:
        raise FileNotFoundError(f"No Prophet model artifact found in {models_dir}/.")


def run_prophet_pipeline(data_path: str = "data/sales.csv"):
    """Execute complete Prophet training workflow."""
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])

    train_df, test_df, _ = chronological_train_test_split(df, train_ratio=0.80)
    model, forecast, meta = train_prophet_model(train_df, test_df)

    print("\n--- Prophet Training Complete ---")
    print(f"Model: {meta['model_type']}")
    print(f"Test Forecast Steps: {meta['test_rows']}")
    print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].head())
    return model, forecast, meta


if __name__ == "__main__":
    run_prophet_pipeline()
