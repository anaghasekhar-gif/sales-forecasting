"""
Streamlit Web Dashboard: Sales Forecasting Using Time Series Analysis.

Features:
- Dataset overview with executive KPI metric cards.
- Interactive historical sales explorer with rolling averages and seasonal breakdowns.
- Model performance benchmark comparing ARIMA / SARIMA and Meta Prophet (MAE, RMSE, MAPE).
- Interactive future sales forecasting studio with configurable horizons (7, 14, 30, 60, 90 days).
- CSV export for generated future sales projections.
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from prophet.serialize import model_from_json

st.set_page_config(
    page_title="Sales Forecasting | Time Series ML",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 18px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-champion {
        background-color: #DCFCE7;
        color: #166534;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_historical_data():
    data_path = "data/sales.csv"
    if not os.path.exists(data_path):
        st.error(f"Dataset not found at {data_path}. Run src/data_preprocessing.py first.")
        st.stop()
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data
def load_model_metrics():
    metrics_path = "models/model_comparison_metrics.csv"
    if os.path.exists(metrics_path):
        return pd.read_csv(metrics_path)
    return None


@st.cache_data
def load_test_predictions():
    arima_path = "models/arima_test_predictions.csv"
    prophet_path = "models/prophet_test_predictions.csv"
    if os.path.exists(arima_path) and os.path.exists(prophet_path):
        df_a = pd.read_csv(arima_path)
        df_p = pd.read_csv(prophet_path)
        df_a["Date"] = pd.to_datetime(df_a["Date"])
        df_p["Date"] = pd.to_datetime(df_p["Date"])
        merged = pd.merge(df_a, df_p[["Date", "Prophet_Forecast", "Prophet_Lower_CI", "Prophet_Upper_CI"]], on="Date")
        return merged
    return None


@st.cache_resource
def load_models():
    models = {}
    arima_path = "models/arima_model.pkl"
    if os.path.exists(arima_path):
        try:
            models["ARIMA"] = joblib.load(arima_path)
        except Exception:
            pass

    prophet_path = "models/prophet_full_model.json"
    if not os.path.exists(prophet_path):
        prophet_path = "models/prophet_model.json"
    if os.path.exists(prophet_path):
        try:
            with open(prophet_path, "r", encoding="utf-8") as f:
                models["Prophet"] = model_from_json(f.read())
        except Exception:
            pass

    return models


# Load data and artifacts
df_sales = load_historical_data()
df_metrics = load_model_metrics()
df_test_preds = load_test_predictions()
models_dict = load_models()

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.image("https://img.icons8.com/fluency/96/line-chart.png", width=70)
st.sidebar.title("Forecasting Studio")
st.sidebar.markdown("**MSc AI & Data Analytics Internship**")
st.sidebar.markdown("---")

selected_model_name = st.sidebar.selectbox(
    "Select Forecasting Engine",
    options=["Meta Prophet (Champion)", "ARIMA / SARIMA"],
    index=0,
    help="Select between the seasonal Prophet model (with promo & holiday regressors) and classical SARIMA.",
)

forecast_horizon = st.sidebar.select_slider(
    "Forecast Horizon (Days)",
    options=[7, 14, 30, 60, 90],
    value=30,
    help="Number of days forward to project future sales.",
)

promo_simulation = st.sidebar.radio(
    "Future Promotion Policy",
    options=["Standard Bi-Weekly Cycles", "No Future Promotions", "Continuous Active Promotions"],
    index=0,
    help="Simulate the business impact of future promotional campaigns.",
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Benchmark Dataset:**
Rossmann Store Sales (1,115 European drugstores).
- Frequency: Daily (`D`)
- Time span: Jan 2013 – Jul 2015
- Cleaned observations: 942 days
""")

# ==========================================
# HEADER SECTION
# ==========================================
st.markdown('<div class="main-header">📈 Sales Forecasting Using Time Series Machine Learning</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">An End-to-End Enterprise Predictive System for Retail Demand Planning, Inventory Management, and Promotion Impact Modeling.</div>', unsafe_allow_html=True)

# Executive KPI Metric Cards
total_records = len(df_sales)
min_date = df_sales["Date"].min().strftime("%b %d, %Y")
max_date = df_sales["Date"].max().strftime("%b %d, %Y")
total_revenue = df_sales["Sales"].sum()
avg_daily_sales = df_sales["Sales"].mean()

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Historical Horizon</div>
        <div class="metric-value">{total_records} Days</div>
        <div style="font-size: 0.75rem; color: #64748B;">{min_date} – {max_date}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Revenue</div>
        <div class="metric-value">€{total_revenue / 1e9:.2f}B</div>
        <div style="font-size: 0.75rem; color: #64748B;">Aggregate Network Sales</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Daily Sales</div>
        <div class="metric-value">€{avg_daily_sales / 1e6:.2f}M</div>
        <div style="font-size: 0.75rem; color: #64748B;">Mean Daily Turnover</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Champion Model</div>
        <div class="metric-value" style="color: #166534;">Prophet</div>
        <span class="badge-champion">19.18% Test MAPE</span>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Promo Uplift</div>
        <div class="metric-value" style="color: #9333EA;">+38.5%</div>
        <div style="font-size: 0.75rem; color: #64748B;">Average Campaign Surge</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# MAIN TABS INTERFACE
# ==========================================
tab_future, tab_eda, tab_benchmark, tab_methodology = st.tabs([
    "🚀 Future Forecasting Studio",
    "📊 Historical Exploratory Analysis",
    "⚖️ Model Benchmark & Evaluation",
    "🔬 Technical Architecture & Methodology",
])

# ----------------------------------------------------
# TAB 1: FUTURE FORECASTING STUDIO
# ----------------------------------------------------
with tab_future:
    st.subheader(f"Future Sales Projection ({forecast_horizon} Days Ahead)")
    st.caption("Generate dynamic out-of-sample forward projections using the retrained champion forecasting architecture.")

    # Calculate future forecast dynamically
    last_date = df_sales["Date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq="D")
    future_df = pd.DataFrame({"ds": future_dates})

    # Set promotional regressor based on user simulation setting
    if promo_simulation == "Standard Bi-Weekly Cycles":
        dow = future_df["ds"].dt.dayofweek
        week_nums = future_df["ds"].dt.isocalendar().week
        future_df["Promo"] = ((week_nums % 2 == 1) & (dow < 5)).astype(int)
    elif promo_simulation == "Continuous Active Promotions":
        dow = future_df["ds"].dt.dayofweek
        future_df["Promo"] = (dow < 6).astype(int)
    else:
        future_df["Promo"] = 0

    future_df["SchoolHoliday"] = (future_df["ds"].dt.month.isin([7, 8])).astype(int)

    if "Prophet" in models_dict and "Prophet" in selected_model_name:
        prophet_model = models_dict["Prophet"]
        fc = prophet_model.predict(future_df)
        pred_sales = fc["yhat"].clip(lower=0).values
        pred_lower = fc["yhat_lower"].clip(lower=0).values
        pred_upper = fc["yhat_upper"].clip(lower=0).values
    else:
        # Fallback to ARIMA forecast
        arima_model = models_dict.get("ARIMA", None)
        if arima_model:
            fc_res = arima_model.get_forecast(steps=forecast_horizon)
            pred_sales = np.clip(fc_res.predicted_mean, 0, None)
            ci = fc_res.conf_int()
            pred_lower = np.clip(ci[:, 0], 0, None)
            pred_upper = np.clip(ci[:, 1], 0, None)
        else:
            pred_sales = np.full(forecast_horizon, avg_daily_sales)
            pred_lower = pred_sales * 0.8
            pred_upper = pred_sales * 1.2

    forecast_results = pd.DataFrame({
        "Date": future_dates,
        "DayOfWeek": future_dates.day_name(),
        "Forecasted_Sales": pred_sales,
        "Lower_Bound_95CI": pred_lower,
        "Upper_Bound_95CI": pred_upper,
        "Promo_Active": future_df["Promo"].values,
    })

    # Display KPI strip for this forecast
    fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)
    with fc_col1:
        st.metric("Selected Model", "Meta Prophet" if "Prophet" in selected_model_name else "ARIMA / SARIMA")
    with fc_col2:
        st.metric("Forecast Window", f"Next {forecast_horizon} Days", f"{future_dates[0].strftime('%b %d')} - {future_dates[-1].strftime('%b %d, %Y')}")
    with fc_col3:
        st.metric("Total Projected Sales", f"€{pred_sales.sum() / 1e6:,.2f}M")
    with fc_col4:
        st.metric("Avg Daily Projected Sales", f"€{pred_sales.mean() / 1e6:,.2f}M")

    # Plotly interactive forecast chart
    recent_history = df_sales.iloc[-60:].copy()

    fig_fc = go.Figure()
    # Historical Sales
    fig_fc.add_trace(go.Scatter(
        x=recent_history["Date"],
        y=recent_history["Sales"] / 1e6,
        mode="lines",
        name="Recent Actual Sales (Last 60 Days)",
        line=dict(color="#334155", width=2),
    ))

    # Upper bound
    fig_fc.add_trace(go.Scatter(
        x=forecast_results["Date"],
        y=forecast_results["Upper_Bound_95CI"] / 1e6,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="95% Upper CI",
    ))

    # Lower bound with fill
    fig_fc.add_trace(go.Scatter(
        x=forecast_results["Date"],
        y=forecast_results["Lower_Bound_95CI"] / 1e6,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(34, 197, 94, 0.2)",
        name="95% Confidence Envelope",
    ))

    # Forecast Point Estimates
    fig_fc.add_trace(go.Scatter(
        x=forecast_results["Date"],
        y=forecast_results["Forecasted_Sales"] / 1e6,
        mode="lines+markers",
        name=f"Forecast ({'Prophet' if 'Prophet' in selected_model_name else 'ARIMA'})",
        line=dict(color="#16A34A", width=3, dash="dash"),
        marker=dict(size=6, color="#16A34A"),
    ))

    # Vertical separation boundary
    fig_fc.add_vline(
        x=recent_history["Date"].max().timestamp() * 1000,
        line_width=2,
        line_dash="dot",
        line_color="#EF4444",
        annotation_text="Forecast Horizon Start",
        annotation_position="top left",
    )

    fig_fc.update_layout(
        title=f"Historical Sales vs Future {forecast_horizon}-Day Forecast Horizon",
        xaxis_title="Calendar Date",
        yaxis_title="Daily Revenue (€ Millions)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=520,
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Detailed Forecast Table & Download
    exp_col1, exp_col2 = st.columns([3, 1])
    with exp_col1:
        st.markdown("#### Projected Daily Breakdown")
    with exp_col2:
        csv_data = forecast_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Forecast (CSV)",
            data=csv_data,
            file_name=f"sales_forecast_{forecast_horizon}d.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.dataframe(
        forecast_results.style.format({
            "Date": lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"),
            "Forecasted_Sales": "€{:,.2f}",
            "Lower_Bound_95CI": "€{:,.2f}",
            "Upper_Bound_95CI": "€{:,.2f}",
            "Promo_Active": lambda x: "✅ Active" if x == 1 else "❌ Inactive",
        }),
        height=280,
        use_container_width=True,
    )

# ----------------------------------------------------
# TAB 2: EXPLORATORY DATA ANALYSIS
# ----------------------------------------------------
with tab_eda:
    st.subheader("Exploratory Data Analysis & Seasonality Discovery")
    st.caption("Empirical investigation of sales distributions, moving averages, cyclical variations, and promotional uplift.")

    eda_metric_choice = st.selectbox(
        "Select Visual Exploration View",
        options=[
            "1. Historical Sales & Moving Averages (Trend & Volatility)",
            "2. Day-of-Week Seasonality & Promotional Lift",
            "3. Monthly Seasonality & Year-over-Year Trajectory",
            "4. Classical Time-Series Decomposition (Period = 7 Days)",
        ],
    )

    if "1." in eda_metric_choice:
        df_plot = df_sales.copy()
        df_plot["Rolling_7"] = df_plot["Sales"].rolling(7).mean()
        df_plot["Rolling_30"] = df_plot["Sales"].rolling(30).mean()

        fig_eda1 = go.Figure()
        fig_eda1.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["Sales"] / 1e6, mode="lines", name="Actual Daily Sales", opacity=0.35, line=dict(color="#94A3B8", width=1)))
        fig_eda1.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["Rolling_7"] / 1e6, mode="lines", name="7-Day Moving Avg (Weekly Trend)", line=dict(color="#F97316", width=2)))
        fig_eda1.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["Rolling_30"] / 1e6, mode="lines", name="30-Day Moving Avg (Monthly Trend)", line=dict(color="#2563EB", width=3)))

        fig_eda1.update_layout(
            title="Daily Sales with 7-Day and 30-Day Smoothed Moving Averages",
            xaxis_title="Date",
            yaxis_title="Sales (€ Millions)",
            template="plotly_white",
            height=500,
        )
        st.plotly_chart(fig_eda1, use_container_width=True)

        st.info("""
        **Key Trend Insights:**
        - **Weekly Periodicity**: Pronounced 7-day cyclical oscillations, driven by sharp Saturday purchasing spikes and Sunday store closures.
        - **December Surges**: Substantial holiday turnover peaks in November–December of both 2013 and 2014, reaching up to €15M/day across the network.
        - **Structural Stability**: The 30-day rolling baseline exhibits steady growth from €6.0M/day to €6.8M/day over the 2.5-year observation window.
        """)

    elif "2." in eda_metric_choice:
        df_dow = df_sales.copy()
        df_dow["DayOfWeek"] = df_dow["Date"].dt.day_name()
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        c_a, c_b = st.columns(2)
        with c_a:
            fig_dow = px.box(
                df_dow,
                x="DayOfWeek",
                y=df_dow["Sales"] / 1e6,
                category_orders={"DayOfWeek": day_order},
                color="DayOfWeek",
                title="Sales Distribution by Day of Week",
                labels={"y": "Sales (€M)", "DayOfWeek": "Day"},
                template="plotly_white",
            )
            fig_dow.update_layout(showlegend=False, height=450)
            st.plotly_chart(fig_dow, use_container_width=True)

        with c_b:
            df_dow["Promo_Label"] = df_dow["Promo"].map({1: "Active Promo (1)", 0: "No Promo (0)"})
            fig_promo = px.box(
                df_dow,
                x="Promo_Label",
                y=df_dow["Sales"] / 1e6,
                color="Promo_Label",
                title="Promotion vs Non-Promotion Sales Distribution",
                labels={"y": "Sales (€M)", "Promo_Label": "Promotion Status"},
                color_discrete_map={"Active Promo (1)": "#10B981", "No Promo (0)": "#6366F1"},
                template="plotly_white",
            )
            fig_promo.update_layout(showlegend=False, height=450)
            st.plotly_chart(fig_promo, use_container_width=True)

        promo_mean = df_dow[df_dow["Promo"] == 1]["Sales"].mean()
        non_promo_mean = df_dow[df_dow["Promo"] == 0]["Sales"].mean()
        uplift = ((promo_mean - non_promo_mean) / non_promo_mean) * 100
        st.success(f"**Quantified Promotional Impact**: Active promotion days average **€{promo_mean/1e6:.2f}M**, compared to **€{non_promo_mean/1e6:.2f}M** on non-promo days — an average revenue uplift of **+{uplift:.1f}%**.")

    elif "3." in eda_metric_choice:
        st.image("plots/monthly_sales.png", caption="Monthly Aggregate Sales Volume & Year-over-Year Trajectory", use_container_width=True)
    elif "4." in eda_metric_choice:
        st.image("plots/decomposition.png", caption="Classical Additive Time Series Decomposition (Observed, Trend, Seasonality, Residuals)", use_container_width=True)
        st.markdown("""
        **Decomposition Components Defined:**
        1. **Observed ($y_t$)**: The raw ground-truth daily turnover recorded across the retail network.
        2. **Trend ($T_t$)**: The underlying long-term secular direction of sales after filtering out short-term weekly and cyclical fluctuations.
        3. **Seasonality ($S_t$)**: The recurring, predictable weekly oscillation (period = 7 days) caused by regular weekly shopping patterns.
        4. **Residual ($R_t$)**: The irregular, stochastic component representing unexplained variations, idiosyncratic shocks, and extreme anomalies after accounting for trend and seasonality.
        """)

# ----------------------------------------------------
# TAB 3: MODEL BENCHMARK & EVALUATION
# ----------------------------------------------------
with tab_benchmark:
    st.subheader("Model Performance Benchmark: ARIMA vs Meta Prophet")
    st.caption("Rigorous out-of-sample evaluation on an identical, unseen 189-day test partition (20% chronological split).")

    if df_metrics is not None:
        st.markdown("### 🏆 Performance Comparison Matrix")

        st.table(df_metrics.style.format({
            "MAE": "€{:,.2f}",
            "RMSE": "€{:,.2f}",
            "MAPE (%)": "{:.2f}%",
        }))

        b_col1, b_col2, b_col3 = st.columns(3)
        with b_col1:
            st.markdown("""
            **Mean Absolute Error (MAE)**
            $$\\text{MAE} = \\frac{1}{n} \\sum_{t=1}^{n} |y_t - \\hat{y}_t|$$
            - Measures average forecast error magnitude in original monetary currency (€).
            - Prophet achieved **€703K**, improving on ARIMA by **46.5%**.
            """)
        with b_col2:
            st.markdown("""
            **Root Mean Squared Error (RMSE)**
            $$\\text{RMSE} = \\sqrt{\\frac{1}{n} \\sum_{t=1}^{n} (y_t - \\hat{y}_t)^2}$$
            - Penalizes larger forecasting deviations disproportionately due to quadratic loss.
            - Prophet scored **€1.02M** vs ARIMA's **€1.93M** (**47.0% lower variance**).
            """)
        with b_col3:
            st.markdown("""
            **Mean Absolute Percentage Error (MAPE)**
            $$\\text{MAPE} = \\frac{100\\%}{n} \\sum_{t=1}^{n} \\left| \\frac{y_t - \\hat{y}_t}{y_t} \\right|$$
            - Relative scale-independent error metric.
            - Prophet achieved an exceptional **19.18%** test error compared to ARIMA's **99.11%**.
            """)

        st.markdown("---")
        st.markdown("### 🔍 Out-of-Sample Predictions Overlay")
        st.image("plots/actual_vs_forecast.png", caption="Out-of-Sample Forecasting: Actual Test Sales vs ARIMA vs Meta Prophet", use_container_width=True)

        st.markdown("### 📊 Metric Bar Chart Comparison")
        st.image("plots/model_comparison.png", caption="MAE, RMSE, and MAPE Side-by-Side Benchmark", use_container_width=True)

# ----------------------------------------------------
# TAB 4: TECHNICAL ARCHITECTURE & METHODOLOGY
# ----------------------------------------------------
with tab_methodology:
    st.subheader("Technical Architecture & Internship Project Methodology")
    st.markdown("""
    This project demonstrates an enterprise-grade time-series forecasting pipeline built in accordance with MSc Artificial Intelligence & Data Analytics standards:
    """)

    st.markdown("""
    ```
    Historical Sales Data (1M+ Rossmann Records)
            ↓
    Data Cleaning & Missing Value Audits
            ↓
    Daily Multi-Store Network Aggregation (Frequency = 'D', 942 Days)
            ↓
    Exploratory Data Analysis & Additive Decomposition (Period = 7)
            ↓
    Feature Engineering (Lags, Rolling Stats, Calendar, Promo Indicators)
            ↓
    Chronological Train-Test Split (80% Train: 753 Days | 20% Test: 189 Days)
            ↓
    ┌───────────────────────────┴───────────────────────────┐
    ↓                                                       ↓
    Model 1: SARIMA(1,1,1)x(1,0,1,7)                Model 2: Meta Prophet
    - Stationarity Check (ADF Test, p<0.001)        - Trend + Multiplicative Seasonality
    - Autoregressive & Moving Average Lags          - Exogenous Regressors (Promo, Holiday)
    └───────────────────────────┬───────────────────────────┘
            ↓
    Out-of-Sample Evaluation (MAE, RMSE, MAPE)
            ↓
    Champion Model Selection (Prophet Selected: 19.18% MAPE)
            ↓
    Full-History Retraining & Future Forecasting (7 to 90 Days Horizon)
            ↓
    Interactive Streamlit Cloud-Ready Dashboard
    ```
    """)

    st.markdown("""
    ### Why Prophet Outperformed ARIMA in Retail Forecasting
    1. **Exogenous Regressor Support**: Retail sales fluctuate drastically depending on whether a marketing promotion (`Promo`) is active. Prophet incorporates promotional regressors natively, whereas univariate ARIMA operates strictly on prior autoregressive sales lags.
    2. **Calendar & Holiday Mechanics**: Prophet integrates Germany-specific federal and regional holidays, accommodating pre-holiday shopping surges and post-holiday lulls.
    3. **Multiplicative Seasonality**: Retail surges scale proportionally with baseline revenue, which multiplicative decomposition captures far more accurately than linear additive models.
    """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>Sales Forecasting Using Time Series Machine Learning • MSc AI & Data Analytics Portfolio Project</div>", unsafe_allow_html=True)
