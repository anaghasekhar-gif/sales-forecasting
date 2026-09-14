"""
Feature Engineering and Chronological Split Module for Time Series Forecasting.

This module provides:
1. Calendar and temporal feature generation:
   - Year, Month, Week of Year, Day of Month, Day of Week, Quarter, Weekend indicator.
2. Promotional and Holiday indicators:
   - Promo flag, SchoolHoliday flag, StateHoliday flag.
3. Lag and Rolling Window features:
   - Lag-1, Lag-7, Lag-14, Lag-30 days.
   - 7-day and 30-day rolling averages and standard deviations.
4. Prophet data transformer:
   - Formats to required 'ds' (date) and 'y' (target) structure with exogenous regressors.
5. Strict Chronological Train-Test Split:
   - 80% Training (earlier time horizon), 20% Testing (most recent period).
   - Explains in detail why random shuffling induces lookahead leakage.
"""

import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract calendar-based features from the Date column.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'Date' column (pd.Timestamp).

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with temporal features.
    """
    df = df.copy()
    if "Date" not in df.columns:
        raise ValueError("DataFrame must contain a 'Date' column.")

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek  # 0 = Monday, 6 = Sunday
    df["Quarter"] = df["Date"].dt.quarter
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

    logger.info("Generated calendar features: Year, Month, Week, Day, DayOfWeek, Quarter, IsWeekend.")
    return df


def create_lag_and_rolling_features(
    df: pd.DataFrame,
    target_col: str = "Sales",
    lags: list = [1, 7, 14, 30],
    windows: list = [7, 30],
) -> pd.DataFrame:
    """
    Create autoregressive lag and rolling window features.
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)

    # Autoregressive lags
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    # Rolling statistics (using shifted series to avoid current-day leakage)
    for window in windows:
        df[f"{target_col}_roll_mean_{window}"] = (
            df[target_col].shift(1).rolling(window=window).mean()
        )
        df[f"{target_col}_roll_std_{window}"] = (
            df[target_col].shift(1).rolling(window=window).std()
        )

    logger.info(f"Created lag features ({lags}) and rolling features ({windows}).")
    return df


def prepare_prophet_dataframe(
    df: pd.DataFrame,
    date_col: str = "Date",
    target_col: str = "Sales",
    regressors: list = ["Promo", "SchoolHoliday", "StateHoliday"],
) -> pd.DataFrame:
    """
    Prepare data in Meta Prophet's expected schema:
    - 'ds': datetime column
    - 'y': target time series
    - Additional exogenous regressors if available.
    """
    prophet_df = pd.DataFrame()
    prophet_df["ds"] = pd.to_datetime(df[date_col])
    prophet_df["y"] = df[target_col].values

    available_regressors = []
    for reg in regressors:
        if reg in df.columns:
            prophet_df[reg] = df[reg].values
            available_regressors.append(reg)

    logger.info(f"Prepared Prophet DataFrame with {len(prophet_df)} records. Regressors: {available_regressors}")
    return prophet_df


def chronological_train_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    date_col: str = "Date",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Split time series strictly chronologically into Training and Testing partitions.

    RATIONALE (Why random split is prohibited in time series):
    1. Temporal Ordering: Time series observations are autocorrelated. Future states depend on past states.
    2. Data Leakage / Lookahead Bias: Random splitting would place future records into the training set,
       allowing the model to peek into future trends, seasonality, and shocks.
    3. Real-World Alignment: In deployment, forecasts are always out-of-sample forward projections.
       A chronological split faithfully simulates forecasting unseen future horizons.
    """
    df_sorted = df.copy().sort_values(date_col).reset_index(drop=True)
    n_total = len(df_sorted)
    split_idx = int(n_total * train_ratio)

    train_df = df_sorted.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[split_idx:].copy().reset_index(drop=True)

    split_metadata = {
        "total_records": n_total,
        "train_records": len(train_df),
        "test_records": len(test_df),
        "train_ratio": train_ratio,
        "test_ratio": round(1.0 - train_ratio, 2),
        "train_start": train_df[date_col].min(),
        "train_end": train_df[date_col].max(),
        "test_start": test_df[date_col].min(),
        "test_end": test_df[date_col].max(),
    }

    logger.info(
        f"Chronological Split: Train [{split_metadata['train_start'].strftime('%Y-%m-%d')} to "
        f"{split_metadata['train_end'].strftime('%Y-%m-%d')}] ({len(train_df)} rows, {split_metadata['train_ratio']*100:.0f}%) | "
        f"Test [{split_metadata['test_start'].strftime('%Y-%m-%d')} to "
        f"{split_metadata['test_end'].strftime('%Y-%m-%d')}] ({len(test_df)} rows, {split_metadata['test_ratio']*100:.0f}%)"
    )

    return train_df, test_df, split_metadata


if __name__ == "__main__":
    sales_df = pd.read_csv("data/sales.csv")
    sales_df["Date"] = pd.to_datetime(sales_df["Date"])
    feat_df = create_calendar_features(sales_df)
    train, test, meta = chronological_train_test_split(feat_df, train_ratio=0.80)
    print("\nChronological Split Metadata:")
    for k, v in meta.items():
        print(f"  {k}: {v}")
