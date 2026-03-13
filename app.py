import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="UPI Forecast Engine", layout="wide")

st.title("UPI Transaction Forecast")

# -----------------------------
# HOLIDAYS (India 2026)
# -----------------------------

HOLIDAYS_2026 = pd.to_datetime([
"2026-01-01","2026-01-14","2026-01-26","2026-02-15","2026-03-04",
"2026-03-19","2026-03-21","2026-03-26","2026-03-31","2026-04-03",
"2026-05-01","2026-05-27","2026-06-26","2026-07-16","2026-08-15",
"2026-08-26","2026-08-28","2026-09-04","2026-09-14","2026-10-02",
"2026-10-20","2026-11-08","2026-11-24","2026-12-25"
])

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.header("Forecast Settings")

daily_projection = st.sidebar.checkbox("Enable Daily Projection")

if daily_projection:

    forecast_days = st.sidebar.slider(
        "Daily Forecast Horizon (Days)",
        7,
        120,
        30
    )

else:

    forecast_months = st.sidebar.slider(
        "Monthly Forecast Horizon (Months)",
        1,
        24,
        6
    )

# -----------------------------
# DATA LOADING
# -----------------------------

@st.cache_data
def load_daily():

    df = pd.read_excel(
        "data/merged_upi_transactions.xlsx",
        engine="openpyxl"
    )

    df.columns = df.columns.str.strip()

    df["DATE"] = pd.to_datetime(df["DATE"])

    df = df.sort_values("DATE")

    return df


@st.cache_data
def load_monthly():

    df = pd.read_excel(
        "data/UPI_Transactions.xlsx",
        engine="openpyxl"
    )

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date")

    df.set_index("Date", inplace=True)

    return df


# -----------------------------
# FORECAST FUNCTION
# -----------------------------

def forecast_series(series, horizon, dates):

    series = series.astype(float)

    # remove extreme outliers
    q_low = series.quantile(0.01)
    q_high = series.quantile(0.99)
    series = series.clip(q_low, q_high)

    smooth = series.ewm(span=8).mean()

    growth = smooth.pct_change().dropna()

    base_growth = growth.rolling(14).mean().dropna()

    avg_growth = base_growth.mean()

    last_actual = series.iloc[-1]

    future_vals = []
    future_dates = []

    # anchor forecast
    future_vals.append(last_actual)
    future_dates.append(dates.iloc[-1])

    prev_val = last_actual

    # deterministic random generator
    rng = np.random.default_rng(42)

    for i in range(1, horizon):

        next_date = dates.iloc[-1] + pd.DateOffset(days=i)

        damping = 1 / (1 + 0.03 * i)

        noise = rng.normal(0, growth.std() * 0.15)

        g = avg_growth * damping + noise

        value = prev_val * (1 + g)

        if next_date in HOLIDAYS_2026:
            value *= 1.03

        future_vals.append(value)
        future_dates.append(next_date)

        prev_val = value

    future_vals = pd.Series(future_vals)

    smoothed = future_vals.copy()
    smoothed.iloc[1:] = future_vals.iloc[1:].rolling(3, min_periods=1).mean()

    return smoothed.values, future_dates


# -----------------------------
# DAILY FORECAST
# -----------------------------

if daily_projection:

    df = load_daily()

    numeric_cols = df.select_dtypes(include=np.number).columns

    field = st.sidebar.selectbox(
        "Select Projection Field",
        numeric_cols
    )

    data = df[["DATE", field]].rename(columns={field: "y"})

    series = data["y"]

    future_vals, future_dates = forecast_series(
        series,
        forecast_days,
        data["DATE"]
    )

    history = series.tail(30)
    history_dates = data["DATE"].tail(30)

# -----------------------------
# MONTHLY FORECAST
# -----------------------------

else:

    df = load_monthly()

    field = st.sidebar.selectbox(
        "Select Projection Field",
        df.columns
    )

    series = df[field].resample("M").sum()

    series = series.ewm(span=6).mean()

    growth = series.pct_change().dropna()

    avg_growth = growth.mean()

    last_val = series.iloc[-1]

    future_vals = []
    future_dates = []

    rng = np.random.default_rng(42)

    for i in range(forecast_months):

        damping = 1 / (1 + 0.2 * i)

        noise = rng.normal(0, growth.std() * 0.3)

        g = avg_growth * damping + noise

        value = last_val * (1 + g)

        future_vals.append(value)

        future_dates.append(
            series.index[-1] + pd.DateOffset(months=i + 1)
        )

        last_val = value

    history = series.tail(30)
    history_dates = history.index


# -----------------------------
# GRAPH
# -----------------------------

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=history_dates,
    y=history.values,
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

# -----------------------------
# FORECAST TABLE
# -----------------------------

last_actual = history.values[-1]

growth_pct = ((future_vals - last_actual) / last_actual) * 100

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast": future_vals,
    "Growth %": growth_pct
})

st.subheader("Forecast Table")

st.dataframe(forecast_df, use_container_width=True)

max_idx = np.argmax(future_vals)

st.write(
    "Maximum projected date:",
    future_dates[max_idx],
    "Projected value:",
    round(future_vals[max_idx], 2)
)
