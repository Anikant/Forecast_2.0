import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="UPI Transaction Forecast Engine", layout="wide")

st.title("UPI Transaction Forecast Engine")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("Forecast Settings")

daily_projection = st.sidebar.checkbox("Enable Daily Projection")

if daily_projection:

    forecast_days = st.sidebar.slider(
        "Daily Forecast Horizon (Days)",
        min_value=7,
        max_value=120,
        value=30
    )

else:

    forecast_months = st.sidebar.slider(
        "Monthly Forecast Horizon (Months)",
        min_value=1,
        max_value=24,
        value=6
    )

# ---------------------------------------------------
# HOLIDAY LIST (INDIA 2026)
# ---------------------------------------------------

def get_india_holidays():

    holidays = pd.to_datetime([
        "2026-01-01",
        "2026-01-14",
        "2026-01-26",
        "2026-02-15",
        "2026-03-04",
        "2026-03-19",
        "2026-03-21",
        "2026-03-26",
        "2026-03-31",
        "2026-04-03",
        "2026-05-01",
        "2026-05-27",
        "2026-06-26",
        "2026-07-16",
        "2026-08-15",
        "2026-08-26",
        "2026-08-28",
        "2026-09-04",
        "2026-09-14",
        "2026-10-02",
        "2026-10-20",
        "2026-11-08",
        "2026-11-24",
        "2026-12-25"
    ])

    return holidays

# ---------------------------------------------------
# OUTLIER REMOVAL
# ---------------------------------------------------

def remove_outliers(series):

    z = (series - series.mean()) / series.std()

    series[z.abs() > 3] = series.rolling(7).median()

    return series

# ---------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------

@st.cache_data
def load_daily():

    df = pd.read_excel("data/merged_upi_transactions.xlsx")

    df.columns = df.columns.str.strip()

    df["DATE"] = pd.to_datetime(df["DATE"])

    df = df.sort_values("DATE")

    return df


@st.cache_data
def load_monthly():

    df = pd.read_excel("data/UPI_Transactions.xlsx")

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date")

    df.set_index("Date", inplace=True)

    return df


# ---------------------------------------------------
# MONTHLY FORECAST
# ---------------------------------------------------

if not daily_projection:

    df = load_monthly()

    field = st.sidebar.selectbox(
        "Select Projection Field",
        df.columns
    )

    series = df[field].resample("M").sum()

    series = remove_outliers(series)

    growth = np.log(series / series.shift(1)).dropna()

    rolling_growth = growth.rolling(6).mean().dropna()

    avg_growth = rolling_growth.mean()

    last_val = series.iloc[-1]

    future_vals = []

    for i in range(forecast_months):

        damping = 1 / (1 + 0.15 * i)

        last_val = last_val * (1 + avg_growth * damping)

        future_vals.append(last_val)

    future_vals = np.array(future_vals)

    future_dates = [
        series.index[-1] + pd.DateOffset(months=i+1)
        for i in range(forecast_months)
    ]

    history = series

# ---------------------------------------------------
# DAILY FORECAST
# ---------------------------------------------------

else:

    df = load_daily()

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    field = st.sidebar.selectbox(
        "Select Projection Field",
        numeric_cols
    )

    data = df[["DATE", field]].rename(columns={field: "y"})

    series = data["y"]

    series = remove_outliers(series)

    growth = np.log(series / series.shift(1)).dropna()

    rolling_growth = growth.rolling(14).mean().dropna()

    avg_growth = rolling_growth.mean()

    last_val = series.iloc[-1]

    holidays = get_india_holidays()

    future_vals = []

    for i in range(forecast_days):

        damping = 1 / (1 + 0.02 * i)

        last_val = last_val * (1 + avg_growth * damping)

        next_date = data["DATE"].iloc[-1] + pd.DateOffset(days=i+1)

        if next_date in holidays:

            last_val = last_val * 1.05

        future_vals.append(last_val)

    future_vals = pd.Series(future_vals).rolling(3, min_periods=1).mean().values

    future_dates = [
        data["DATE"].iloc[-1] + pd.DateOffset(days=i+1)
        for i in range(forecast_days)
    ]

    history = series


# ---------------------------------------------------
# GRAPH
# ---------------------------------------------------

fig = go.Figure()

recent = history.tail(30)

fig.add_trace(go.Scatter(
    x=recent.index if not daily_projection else df["DATE"].tail(30),
    y=recent.values,
    mode="lines",
    name="Actual"
))

fig.add_trace(go.Scatter(
    x=future_dates,
    y=future_vals,
    mode="lines+markers",
    name="Forecast"
))

fig.update_layout(
    template="plotly_white",
    height=550,
    xaxis_title="Date",
    yaxis_title="Transaction Volume"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# FORECAST TABLE
# ---------------------------------------------------

last_actual = history.iloc[-1]

growth_pct = ((future_vals - last_actual) / last_actual) * 100

forecast_table = pd.DataFrame({
    "Date": future_dates,
    "Forecast": future_vals,
    "Growth %": growth_pct
})

st.markdown("### Forecast Table")

st.dataframe(forecast_table, use_container_width=True)

max_idx = np.argmax(future_vals)

st.write(
    "Maximum projected day:",
    future_dates[max_idx],
    "Projected value:",
    round(future_vals[max_idx], 2)
)
