import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="UPI Forecast Engine", layout="wide")

st.title("UPI Transaction Forecast")

# -------------------------
# HOLIDAYS INDIA 2026
# -------------------------

HOLIDAYS_2026 = pd.to_datetime([
"2026-01-01","2026-01-14","2026-01-26","2026-02-15","2026-03-04",
"2026-03-19","2026-03-21","2026-03-26","2026-03-31","2026-04-03",
"2026-05-01","2026-05-27","2026-06-26","2026-07-16","2026-08-15",
"2026-08-26","2026-08-28","2026-09-04","2026-09-14","2026-10-02",
"2026-10-20","2026-11-08","2026-11-24","2026-12-25"
])

# -------------------------
# SIDEBAR
# -------------------------

st.sidebar.header("Projection Settings")

mode = st.sidebar.radio(
    "Projection Mode",
    ["Daily","Monthly"]
)

days = st.sidebar.slider(
    "Forecast Horizon (Days)",
    30,
    365,
    120
)

# -------------------------
# LOAD DATA
# -------------------------

@st.cache_data
def load_data():

    df = pd.read_excel(
        "data/merged_upi_transactions.xlsx",
        engine="openpyxl"
    )

    df.columns = df.columns.str.strip()

    df["DATE"] = pd.to_datetime(df["DATE"])

    df = df.sort_values("DATE")

    return df


df = load_data()

numeric_cols = df.select_dtypes(include=np.number).columns

field = st.sidebar.selectbox(
    "Select Projection Field",
    numeric_cols
)

series = df[field]
dates = df["DATE"]

# -------------------------
# FORECAST ENGINE
# -------------------------

def forecast_daily(series, horizon, dates):

    series = series.astype(float)

    # smooth trend
    smooth = series.ewm(span=10).mean()

    growth = smooth.pct_change().dropna()

    avg_growth = growth.tail(30).mean()

    last_actual = series.iloc[-1]

    future_vals = []
    future_dates = []

    future_vals.append(last_actual)
    future_dates.append(dates.iloc[-1])

    prev_val = last_actual

    for i in range(1, horizon):

        next_date = dates.iloc[-1] + pd.DateOffset(days=i)

        # damping to prevent exponential explosion
        damping = 1/(1 + 0.025*i)

        noise = np.random.normal(0, growth.std()*0.1)

        g = avg_growth*damping + noise

        value = prev_val*(1+g)

        # holiday adjustment
        if next_date in HOLIDAYS_2026:
            value *= 1.02

        future_vals.append(value)
        future_dates.append(next_date)

        prev_val = value

    return pd.Series(future_vals, index=future_dates)


forecast_series = forecast_daily(series, days, dates)

# -------------------------
# DAILY VIEW
# -------------------------

if mode == "Daily":

    history = series.tail(30)
    history_dates = dates.tail(30)

    future_vals = forecast_series.values
    future_dates = forecast_series.index

# -------------------------
# MONTHLY VIEW
# -------------------------

else:

    full_series = pd.concat([
        pd.Series(series.values,index=dates),
        forecast_series.iloc[1:]
    ])

    monthly = full_series.resample("M").sum()

    history = monthly.iloc[-6:-1]
    future = monthly.iloc[-1:]

    history_dates = history.index
    future_dates = future.index

    future_vals = future.values

# -------------------------
# GRAPH
# -------------------------

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
    yaxis_title="Value"
)

st.plotly_chart(fig,use_container_width=True)

# -------------------------
# TABLE
# -------------------------

last_actual = history.values[-1]

growth_pct = ((future_vals-last_actual)/last_actual)*100

table = pd.DataFrame({
"Date":future_dates,
"Forecast":future_vals,
"Growth %":growth_pct
})

st.subheader("Projection Table")

st.dataframe(table,use_container_width=True)
