"""
Exploratory Data Analysis & Time Series Decomposition Module.

This module performs:
1. Distribution and summary statistics analysis (mean, median, std, skewness, kurtosis).
2. Outlier detection using the Interquartile Range (IQR) rule.
3. Trend & rolling statistics (7-day and 30-day moving averages and volatility bands).
4. Calendar & seasonal pattern extraction (day-of-week, monthly, year-over-year).
5. Promotional and holiday uplift analysis.
6. Classical time-series decomposition into Observed, Trend, Seasonality, and Residuals.
7. Generation of publication-quality plots saved in the 'plots/' directory:
   - plots/sales_over_time.png
   - plots/monthly_sales.png
   - plots/rolling_average.png
   - plots/decomposition.png
   - plots/seasonal_analysis.png
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Aesthetic styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "figure.autolayout": True,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
})

COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "accent": "#2ca02c",
    "dark": "#2c3e50",
    "light": "#ecf0f1",
    "danger": "#d62728",
    "promo": "#9467bd",
}


def load_clean_sales(filepath: str = "data/sales.csv") -> pd.DataFrame:
    """Load the preprocessed sales dataset and ensure datetime index."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing '{filepath}'. Please run data_preprocessing.py first.")
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def compute_summary_statistics(df: pd.DataFrame) -> dict:
    """Compute central tendency, dispersion, shape, and outlier statistics."""
    sales = df["Sales"]
    q1 = sales.quantile(0.25)
    q3 = sales.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(sales < lower_bound) | (sales > upper_bound)]

    stats = {
        "count": len(sales),
        "mean": sales.mean(),
        "median": sales.median(),
        "std": sales.std(),
        "min": sales.min(),
        "max": sales.max(),
        "skewness": sales.skew(),
        "kurtosis": sales.kurtosis(),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_outlier_bound": lower_bound,
        "upper_outlier_bound": upper_bound,
        "outlier_count": len(outliers),
        "outlier_percentage": (len(outliers) / len(sales)) * 100,
    }
    return stats


def plot_sales_over_time(df: pd.DataFrame, output_dir: str = "plots"):
    """Generate and save historical sales line chart."""
    logger.info("Generating plot: sales_over_time.png")
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)

    ax.plot(df["Date"], df["Sales"] / 1e6, color=COLORS["primary"], linewidth=1.2, label="Daily Total Sales")

    # Add trend line using polynomial regression
    x_num = np.arange(len(df))
    poly_fit = np.polyfit(x_num, df["Sales"] / 1e6, deg=2)
    poly_trend = np.poly1d(poly_fit)
    ax.plot(df["Date"], poly_trend(x_num), color=COLORS["danger"], linestyle="--", linewidth=2.0, label="Polynomial Trend (Degree 2)")

    ax.set_title("Historical Daily Sales Performance (2013 - 2015)", fontweight="bold", pad=15)
    ax.set_xlabel("Date", labelpad=10)
    ax.set_ylabel("Total Sales (€ Millions)", labelpad=10)
    ax.legend(frameon=True, facecolor="white", loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)

    save_path = os.path.join(output_dir, "sales_over_time.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_monthly_sales(df: pd.DataFrame, output_dir: str = "plots"):
    """Generate and save monthly aggregated sales and year-over-year comparison."""
    logger.info("Generating plot: monthly_sales.png")
    df_copy = df.copy()
    df_copy["Year"] = df_copy["Date"].dt.year
    df_copy["Month"] = df_copy["Date"].dt.month
    df_copy["YearMonth"] = df_copy["Date"].dt.to_period("M").dt.to_timestamp()

    monthly = df_copy.groupby("YearMonth")["Sales"].sum().reset_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=300, sharex=False)

    # 1. Total monthly sales trend bar chart
    sns.barplot(
        data=monthly,
        x=monthly["YearMonth"].dt.strftime("%b %Y"),
        y=monthly["Sales"] / 1e6,
        ax=ax1,
        color=COLORS["primary"],
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_title("Total Monthly Sales Volume (€ Millions)", fontweight="bold")
    ax1.set_xlabel("Month-Year")
    ax1.set_ylabel("Sales (€ Millions)")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # 2. Year-over-Year monthly profile
    yoy = df_copy.groupby(["Year", "Month"])["Sales"].sum().reset_index()
    palette = {2013: "#1f77b4", 2014: "#2ca02c", 2015: "#d62728"}
    for year in yoy["Year"].unique():
        sub = yoy[yoy["Year"] == year]
        ax2.plot(
            sub["Month"],
            sub["Sales"] / 1e6,
            marker="o",
            linewidth=2.2,
            label=f"Year {year}",
            color=palette.get(year, "#7f7f7f"),
        )
    ax2.set_title("Year-over-Year Monthly Sales Seasonality Comparison", fontweight="bold")
    ax2.set_xlabel("Month of Year (1 = Jan, 12 = Dec)")
    ax2.set_ylabel("Sales (€ Millions)")
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax2.legend(title="Calendar Year", frameon=True, facecolor="white")
    ax2.grid(True, linestyle=":", alpha=0.5)

    save_path = os.path.join(output_dir, "monthly_sales.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_rolling_average(df: pd.DataFrame, output_dir: str = "plots"):
    """Generate and save rolling mean (7-day, 30-day) and rolling volatility plot."""
    logger.info("Generating plot: rolling_average.png")
    df_copy = df.copy().sort_values("Date")
    df_copy["Rolling_7"] = df_copy["Sales"].rolling(window=7, center=False).mean()
    df_copy["Rolling_30"] = df_copy["Sales"].rolling(window=30, center=False).mean()
    df_copy["Rolling_Std_30"] = df_copy["Sales"].rolling(window=30, center=False).std()

    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)

    ax.plot(df_copy["Date"], df_copy["Sales"] / 1e6, color="#bdc3c7", alpha=0.45, linewidth=0.9, label="Daily Actual Sales")
    ax.plot(df_copy["Date"], df_copy["Rolling_7"] / 1e6, color=COLORS["secondary"], linewidth=1.5, label="7-Day Rolling Mean (Weekly)")
    ax.plot(df_copy["Date"], df_copy["Rolling_30"] / 1e6, color=COLORS["primary"], linewidth=2.2, label="30-Day Rolling Mean (Monthly Trend)")

    # 30-Day Volatility Ribbon
    upper_band = (df_copy["Rolling_30"] + 1.96 * df_copy["Rolling_Std_30"]) / 1e6
    lower_band = (df_copy["Rolling_30"] - 1.96 * df_copy["Rolling_Std_30"]) / 1e6
    ax.fill_between(df_copy["Date"], lower_band, upper_band, color=COLORS["primary"], alpha=0.15, label="±1.96 Rolling Volatility Band (30-Day)")

    ax.set_title("Rolling Trend & Volatility Analysis (Smoothing Window)", fontweight="bold", pad=15)
    ax.set_xlabel("Date", labelpad=10)
    ax.set_ylabel("Sales (€ Millions)", labelpad=10)
    ax.legend(frameon=True, facecolor="white", loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)

    save_path = os.path.join(output_dir, "rolling_average.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_decomposition(df: pd.DataFrame, output_dir: str = "plots"):
    """Perform additive time-series decomposition (Observed, Trend, Seasonality, Residuals)."""
    logger.info("Generating plot: decomposition.png")
    df_ts = df.set_index("Date")["Sales"].asfreq("D")
    
    # 7-day period captures day-of-week retail seasonality
    decomp = seasonal_decompose(df_ts, model="additive", period=7, extrapolate_trend="freq")

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), dpi=300, sharex=True)

    axes[0].plot(decomp.observed / 1e6, color=COLORS["dark"], linewidth=1.2)
    axes[0].set_ylabel("Observed (€M)")
    axes[0].set_title("Time Series Classical Decomposition (Additive, Period = 7 Days)", fontweight="bold", pad=12)

    axes[1].plot(decomp.trend / 1e6, color=COLORS["primary"], linewidth=2.0)
    axes[1].set_ylabel("Trend (€M)")

    axes[2].plot(decomp.seasonal / 1e6, color=COLORS["accent"], linewidth=1.2)
    axes[2].set_ylabel("Seasonality (€M)")

    axes[3].scatter(decomp.resid.index, decomp.resid / 1e6, color=COLORS["danger"], s=10, alpha=0.65)
    axes[3].axhline(0, color="black", linestyle="--", linewidth=1.0)
    axes[3].set_ylabel("Residuals (€M)")
    axes[3].set_xlabel("Date")

    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.5)

    save_path = os.path.join(output_dir, "decomposition.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_seasonal_and_promotional_analysis(df: pd.DataFrame, output_dir: str = "plots"):
    """Generate multi-panel seasonal, day-of-week, and promotional impact boxplots."""
    logger.info("Generating plot: seasonal_analysis.png")
    df_copy = df.copy()
    df_copy["DayOfWeekName"] = df_copy["Date"].dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # 1. Day of Week Boxplot
    sns.boxplot(
        data=df_copy,
        x="DayOfWeekName",
        y=df_copy["Sales"] / 1e6,
        order=day_order,
        palette="Blues_r",
        ax=ax1,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "red", "markersize": 6},
    )
    ax1.set_title("Sales Distribution by Day of Week", fontweight="bold")
    ax1.set_xlabel("Day of Week")
    ax1.set_ylabel("Sales (€ Millions)")
    ax1.tick_params(axis="x", rotation=30)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # 2. Promotion Impact Comparison
    if "Promo" in df_copy.columns:
        df_copy["Promo_Label"] = df_copy["Promo"].map({1: "Active Promotion (Promo=1)", 0: "No Promotion (Promo=0)"})
        sns.boxplot(
            data=df_copy,
            x="Promo_Label",
            y=df_copy["Sales"] / 1e6,
            palette=["#3498db", "#e74c3c"],
            ax=ax2,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "yellow", "markeredgecolor": "black", "markersize": 7},
        )
        promo_mean = df_copy[df_copy["Promo"] == 1]["Sales"].mean()
        non_promo_mean = df_copy[df_copy["Promo"] == 0]["Sales"].mean()
        uplift_pct = ((promo_mean - non_promo_mean) / non_promo_mean) * 100
        ax2.set_title(f"Promotion Impact: +{uplift_pct:.1f}% Average Uplift", fontweight="bold")
        ax2.set_xlabel("Promotion Status")
        ax2.set_ylabel("Sales (€ Millions)")
        ax2.grid(True, linestyle=":", alpha=0.5)
    else:
        # Fallback distribution
        sns.histplot(df_copy["Sales"] / 1e6, kde=True, ax=ax2, color=COLORS["primary"])
        ax2.set_title("Overall Sales Density & Distribution", fontweight="bold")
        ax2.set_xlabel("Sales (€ Millions)")

    save_path = os.path.join(output_dir, "seasonal_analysis.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def run_full_eda(filepath: str = "data/sales.csv", output_dir: str = "plots") -> dict:
    """Run full EDA and generate all plots."""
    os.makedirs(output_dir, exist_ok=True)
    df = load_clean_sales(filepath)

    stats = compute_summary_statistics(df)
    print("\n" + "=" * 60)
    print("      EXPLORATORY DATA ANALYSIS: STATISTICAL SUMMARY")
    print("=" * 60)
    print(f"Observations:              {stats['count']:,} days")
    print(f"Mean Sales:                €{stats['mean']:,.2f}")
    print(f"Median Sales:              €{stats['median']:,.2f}")
    print(f"Standard Deviation:        €{stats['std']:,.2f}")
    print(f"Min Sales:                 €{stats['min']:,.2f}")
    print(f"Max Sales:                 €{stats['max']:,.2f}")
    print(f"Skewness:                  {stats['skewness']:.3f} ({'Right-skewed' if stats['skewness'] > 0 else 'Left-skewed'})")
    print(f"Kurtosis:                  {stats['kurtosis']:.3f}")
    print(f"IQR Outlier Count:         {stats['outlier_count']:,} ({stats['outlier_percentage']:.2f}%)")

    # Generate all required plots
    plot_sales_over_time(df, output_dir)
    plot_monthly_sales(df, output_dir)
    plot_rolling_average(df, output_dir)
    plot_decomposition(df, output_dir)
    plot_seasonal_and_promotional_analysis(df, output_dir)

    print("\nAll EDA plots saved successfully to:", output_dir)
    return stats


if __name__ == "__main__":
    run_full_eda()
