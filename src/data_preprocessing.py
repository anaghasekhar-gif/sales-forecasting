"""
Data Preprocessing Module for Sales Forecasting.

This module handles:
1. Loading raw sales records (Rossmann Store Sales or pre-aggregated sales.csv).
2. Data quality audits (shape, data types, missing values, duplicates, invalid sales values).
3. Datetime conversion and chronological sorting.
4. Filtering invalid/negative sales and closed store transactions.
5. Multi-store aggregation to produce a unified daily time-series.
6. Frequency validation (ensuring daily frequency 'D' with zero gaps).
7. Exporting the cleaned, ready-to-forecast dataset to data/sales.csv.
"""

import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def audit_data(df: pd.DataFrame, dataset_name: str = "Dataset") -> dict:
    """
    Perform a comprehensive data quality audit on a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    dataset_name : str
        Label for reporting.

    Returns
    -------
    dict
        Audit summary statistics.
    """
    logger.info(f"=== Running Data Audit for {dataset_name} ===")
    n_rows, n_cols = df.shape
    columns = list(df.columns)
    dtypes = df.dtypes.to_dict()
    missing_vals = df.isnull().sum().to_dict()
    total_missing = df.isnull().sum().sum()
    n_duplicates = df.duplicated().sum()

    print(f"\n--- Data Audit: {dataset_name} ---")
    print(f"Total Rows: {n_rows:,}")
    print(f"Total Columns: {n_cols}")
    print(f"Columns: {columns}")
    print(f"Duplicate Rows: {n_duplicates:,}")
    print(f"Total Missing Values: {total_missing:,}")

    print("\nMissing Values Breakdown:")
    for col, count in missing_vals.items():
        pct = (count / n_rows) * 100 if n_rows > 0 else 0
        if count > 0:
            print(f"  - {col}: {count:,} ({pct:.2f}%)")
    if total_missing == 0:
        print("  None detected.")

    audit_summary = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": columns,
        "dtypes": dtypes,
        "missing_vals": missing_vals,
        "n_duplicates": n_duplicates,
    }
    return audit_summary


def load_raw_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Load sales data from either 'sales.csv' or 'train.csv'.
    If 'train.csv' exists, it takes precedence for raw processing.
    """
    sales_path = os.path.join(data_dir, "sales.csv")
    train_path = os.path.join(data_dir, "train.csv")

    if os.path.exists(train_path):
        logger.info(f"Found raw store sales file: {train_path}")
        df = pd.read_csv(train_path, low_memory=False)
        audit_data(df, "Raw Rossmann Store Transactions (train.csv)")
        return df
    elif os.path.exists(sales_path):
        logger.info(f"Found pre-existing sales file: {sales_path}")
        df = pd.read_csv(sales_path)
        audit_data(df, "Existing sales.csv")
        return df
    else:
        raise FileNotFoundError(
            f"No suitable sales file found in {data_dir}/. Expected 'train.csv' or 'sales.csv'."
        )


def clean_and_aggregate_sales(
    df: pd.DataFrame,
    output_path: str = "data/sales.csv",
    save: bool = True,
) -> pd.DataFrame:
    """
    Process raw sales data into a continuous daily time-series.

    Steps:
    1. Parse 'Date' column to datetime and sort chronologically.
    2. Handle duplicates and negative or NaN sales values.
    3. Filter out closed-store days (Open == 0) if store-level data.
    4. Aggregate sales, promotions, and holidays by Date across all stores.
    5. Ensure a strict, continuous daily frequency ('D').
    6. Export to output_path.
    """
    logger.info("Starting data cleaning and aggregation pipeline...")

    # Standardize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Check if 'Date' exists
    date_col = None
    for candidate in ["Date", "date", "ds", "Timestamp"]:
        if candidate in df.columns:
            date_col = candidate
            break
    if not date_col:
        raise ValueError(f"Could not identify a Date column in: {list(df.columns)}")

    # Check if 'Sales' exists
    sales_col = None
    for candidate in ["Sales", "sales", "Weekly_Sales", "Revenue", "y"]:
        if candidate in df.columns:
            sales_col = candidate
            break
    if not sales_col:
        raise ValueError(f"Could not identify a Sales column in: {list(df.columns)}")

    # 1. Convert Date to datetime format
    logger.info(f"Parsing '{date_col}' into datetime format...")
    df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    invalid_dates = df["Date"].isnull().sum()
    if invalid_dates > 0:
        logger.warning(f"Dropping {invalid_dates} rows with unparseable dates.")
        df = df.dropna(subset=["Date"])

    # 2. Chronological sorting
    df = df.sort_values("Date").reset_index(drop=True)

    # 3. Handle duplicates
    initial_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped_dups = initial_rows - len(df)
    if dropped_dups > 0:
        logger.info(f"Removed {dropped_dups:,} duplicate rows.")

    # 4. Check for invalid or negative sales
    invalid_sales = (df[sales_col] < 0).sum()
    if invalid_sales > 0:
        logger.warning(f"Found {invalid_sales} negative sales records. Clipping to 0.")
        df[sales_col] = df[sales_col].clip(lower=0)

    # 5. Check multi-store granularity vs pre-aggregated
    is_multi_store = "Store" in df.columns or "store" in df.columns

    if is_multi_store:
        logger.info("Dataset contains multi-store granularity. Aggregating to network-level daily series.")
        # Filter for open stores if Open column is available
        if "Open" in df.columns:
            open_records = df[df["Open"] == 1]
            logger.info(
                f"Filtered for open stores: {len(open_records):,} open records out of {len(df):,} total."
            )
        else:
            open_records = df

        # Build aggregation dictionary
        agg_dict = {sales_col: "sum"}
        if "Customers" in open_records.columns:
            agg_dict["Customers"] = "sum"
        if "Promo" in open_records.columns:
            # Promo active if at least 50% of stores had active promotion
            agg_dict["Promo"] = lambda x: int((x > 0).mean() >= 0.5)
        if "SchoolHoliday" in open_records.columns:
            agg_dict["SchoolHoliday"] = lambda x: int((x > 0).mean() >= 0.5)
        if "StateHoliday" in open_records.columns:
            agg_dict["StateHoliday"] = lambda x: int((x.astype(str) != "0").mean() > 0.05)

        daily = open_records.groupby("Date").agg(agg_dict).reset_index()
        daily = daily.rename(columns={sales_col: "Sales"})
    else:
        logger.info("Dataset is already at time-series level. Retaining daily structure.")
        daily = df.copy()
        if sales_col != "Sales":
            daily = daily.rename(columns={sales_col: "Sales"})

    # 6. Ensure continuous daily frequency ('D')
    daily = daily.sort_values("Date").reset_index(drop=True)
    min_date = daily["Date"].min()
    max_date = daily["Date"].max()
    full_date_range = pd.date_range(start=min_date, end=max_date, freq="D")
    
    daily = daily.set_index("Date").reindex(full_date_range)
    daily.index.name = "Date"

    # Forward fill or interpolate if any dates were missing
    missing_days = daily["Sales"].isnull().sum()
    if missing_days > 0:
        logger.info(f"Reindexed daily calendar: imputing {missing_days} missing calendar days using forward-fill.")
        daily["Sales"] = daily["Sales"].ffill().bfill()
        for c in daily.columns:
            if c != "Sales":
                daily[c] = daily[c].fillna(0)

    clean_df = daily.reset_index()

    # Final validation audit
    audit_data(clean_df, "Clean Daily Time Series (data/sales.csv)")

    if save:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        clean_df.to_csv(output_path, index=False)
        logger.info(f"Clean daily sales saved successfully to '{output_path}'.")

    return clean_df


if __name__ == "__main__":
    print("=" * 70)
    print("   SALES FORECASTING: DATA PREPROCESSING PIPELINE")
    print("=" * 70)
    raw_data = load_raw_data()
    clean_sales = clean_and_aggregate_sales(raw_data)
    print(f"\nFinal Processed Data Summary:")
    print(f"Date Range: {clean_sales['Date'].min().strftime('%Y-%m-%d')} to {clean_sales['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Total Observations (Days): {len(clean_sales):,}")
    print(f"Average Daily Sales: €{clean_sales['Sales'].mean():,.2f}")
    print(f"Total Revenue: €{clean_sales['Sales'].sum():,.2f}")
    print(clean_sales.head())
