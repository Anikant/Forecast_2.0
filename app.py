import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from prophet import Prophet

st.set_page_config(page_title="UPI Forecast Engine", layout="wide")

st.title("UPI Transaction Forecast Engine")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

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

# --------------------------------------------------
# INDIA HOLIDAYS 2026
# --------------------------------------------------

def get_holidays():

    holidays = pd.DataFrame({

        "holiday":[
        "new_year",
        "makar_sankranti",
        "republic_day",
        "maha_shivratri",
        "holi",
        "ugadi",
        "eid_ul_fitr",
        "ram_navami",
        "mahavir_jayanti",
        "good_friday",
        "buddha_purnima",
        "labour_day",
        "bakri_eid",
        "muharram",
        "rath_yatra",
        "independence_day",
        "onam",
        "milad_un_nabi",
        "raksha_bandhan",
        "janmashtami",
        "vinayaka_chaturthi",
        "gandhi_jayanti",
        "dussehra",
        "diwali",
        "guru_nanak_jayanti",
        "christmas"
        ],

        "ds":pd.to_datetime([

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
        "2026-05-01",
        "2026-05-27",
        "2026-06-26",
        "2026-07-16",
        "2026-08-15",
        "2026-08-26",
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
    })

    return holidays


# --------------------------------------------------
# OUTLIER FILTER
# --------------------------------------------------

def remove_outliers(series):

    z = (series-series.mean())/series.std()

    series[z.abs()>3] = series.rolling(7).median()

    return series


# --------------------------------------------------
# DATA LOAD
# --------------------------------------------------

@st.cache_data
def load_daily():

    df = pd.read_excel("data/merged_upi_transactions.xlsx")

    df.columns = df.columns.str.strip()

    df["DATE"] = pd.to_datetime(df["DATE"])

    df.rename(columns={"DATE":"ds"}, inplace=True)

    df = df.sort_values("ds")

    return df


@st.cache_data
def load_monthly():

    df = pd.read_excel("data/UPI_Transactions.xlsx")

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date")

    df.set_index("Date", inplace=True)

    return df


# --------------------------------------------------
# MONTHLY FORECAST
# --------------------------------------------------

if not daily_projection:

    df = load_monthly()

    field = st.sidebar.selectbox("Projection Field", df.columns)

    series = df[field].resample("M").sum()

    series = remove_outliers(series)

    growth = np.log(series/series.shift(1)).dropna()

    avg_growth = growth.mean()*0.6

    last_val = series.iloc[-1]

    preds = []

    for i in range(forecast_months):

        last_val = last_val*(1+avg_growth)

        preds.append(last_val)

    future_vals = np.array(preds)

    future_dates = [
        series.index[-1]+pd.DateOffset(months=i+1)
        for i in range(forecast_months)
    ]

    history = series


# --------------------------------------------------
# DAILY FORECAST
# --------------------------------------------------

else:

    df = load_daily()

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    field = st.sidebar.selectbox("Projection Field", numeric_cols)

    data = df[["ds",field]].rename(columns={field:"y"})

    data["y"] = remove_outliers(data["y"])

    # prophet model

    prophet = Prophet(
        holidays=get_holidays(),
        changepoint_prior_scale=0.02,
        seasonality_prior_scale=5,
        weekly_seasonality=True,
        yearly_seasonality=True
    )

    prophet.fit(data)

    future = prophet.make_future_dataframe(periods=forecast_days)

    forecast = prophet.predict(future)

    prophet_vals = forecast["yhat"].tail(forecast_days).values

    # growth model

    series = data["y"]

    growth = np.log(series/series.shift(1)).dropna()

    avg_growth = growth.mean()*0.5

    last_val = series.iloc[-1]

    trend_vals=[]

    for i in range(forecast_days):

        damping = 1/(1+0.015*i)

        last_val = last_val*(1+avg_growth*damping)

        trend_vals.append(last_val)

    # ensemble

    future_vals = 0.6*prophet_vals + 0.4*np.array(trend_vals)

    future_vals = pd.Series(future_vals).rolling(3,min_periods=1).mean().values

    future_dates = forecast["ds"].tail(forecast_days)

    history = data.set_index("ds")["y"]


# --------------------------------------------------
# GRAPH
# --------------------------------------------------

fig = go.Figure()

recent = history.tail(30)

fig.add_trace(go.Scatter(
    x=recent.index,
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
    yaxis_title="Transactions"
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# FORECAST TABLE
# --------------------------------------------------

last_actual = history.iloc[-1]

growth_pct = ((future_vals-last_actual)/last_actual)*100

table = pd.DataFrame({

    "Date":future_dates,
    "Forecast":future_vals,
    "Growth %":growth_pct

})

st.dataframe(table,use_container_width=True)

max_idx = np.argmax(future_vals)

st.write(
    f"Maximum projected day: {future_dates.iloc[max_idx].date()} | Value: {round(future_vals[max_idx],2)}"
)
