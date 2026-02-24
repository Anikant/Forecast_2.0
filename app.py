import os
import random
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================
# 🔒 DETERMINISM
# =====================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except:
    pass

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(page_title="UPI Forecast Engine", layout="wide")

st.markdown("""
<h1 style='text-align: center; color: #1F4E79;'>
UPI Forecasting Engine (Monthly + Daily LSTM)
</h1>
""", unsafe_allow_html=True)

st.markdown("---")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    file_path = "data/merged_upi_transactions.xlsx"

    if not os.path.exists(file_path):
        st.error("merged_upi_transactions.xlsx not found in data folder.")
        st.stop()

    df = pd.read_excel(file_path, engine="openpyxl")

    df["Month"] = pd.to_datetime(df["Month"])
    df = df.sort_values("Month")
    df.set_index("Month", inplace=True)

    return df

df = load_data()

# =====================================================
# SIDEBAR OPTIONS
# =====================================================
st.sidebar.header("Projection Mode")

projection_mode = st.sidebar.radio(
    "Select Projection Type",
    ["Monthly Projection", "Daily Projection"]
)

forecast_months = st.sidebar.slider(
    "Forecast Horizon (Months)",
    min_value=1,
    max_value=24,
    value=6
)

# =====================================================
# COMMON DATA PREP
# =====================================================
target_col = "Total Upi Transaction"
data = df[[target_col]].dropna()

data_log = np.log1p(data)
data_diff = data_log.diff().dropna()

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_diff)

lookback = 12

def create_sequences(dataset, lookback):
    X, y = [], []
    for i in range(lookback, len(dataset)):
        X.append(dataset[i-lookback:i])
        y.append(dataset[i])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, lookback)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =====================================================
# MODEL TRAIN FUNCTION
# =====================================================
@st.cache_resource
def train_lstm(X_train, y_train, X_test, y_test):

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        epochs=120,
        batch_size=8,
        validation_data=(X_test, y_test),
        shuffle=False,
        callbacks=[early_stop],
        verbose=0
    )

    return model

with st.spinner("Training LSTM model..."):
    model = train_lstm(X_train, y_train, X_test, y_test)

# =====================================================
# MONTHLY FORECAST
# =====================================================
def monthly_forecast():

    last_sequence = scaled_data[-lookback:]
    current_sequence = last_sequence.reshape(1, lookback, 1)

    future_forecast = []

    for _ in range(forecast_months):
        next_pred = model.predict(current_sequence, verbose=0)[0]
        future_forecast.append(next_pred)

        current_sequence = np.append(
            current_sequence[:, 1:, :],
            [[next_pred]],
            axis=1
        )

    future_forecast = scaler.inverse_transform(future_forecast)

    last_log_value = data_log.iloc[-1].values[0]
    future_values = []

    current_value = last_log_value
    for diff in future_forecast:
        current_value += diff[0]
        future_values.append(current_value)

    future_values = np.expm1(future_values)

    last_date = df.index[-1]

    future_dates = [
        last_date + pd.DateOffset(months=i+1)
        for i in range(forecast_months)
    ]

    return pd.DataFrame({
        "Date": future_dates,
        "Forecast": future_values
    })

# =====================================================
# DAILY FORECAST MODEL
# =====================================================
def daily_projection(monthly_df):

    daily_records = []

    for _, row in monthly_df.iterrows():

        month_total = row["Forecast"]
        month_date = row["Date"]

        days_in_month = month_date.days_in_month
        daily_value = month_total / days_in_month

        for day in range(1, days_in_month + 1):
            daily_records.append({
                "Date": pd.Timestamp(
                    year=month_date.year,
                    month=month_date.month,
                    day=day
                ),
                "Forecast": daily_value
            })

    return pd.DataFrame(daily_records)

# =====================================================
# RUN PROJECTION
# =====================================================
monthly_df = monthly_forecast()

if projection_mode == "Monthly Projection":
    forecast_df = monthly_df
else:
    forecast_df = daily_projection(monthly_df)

# =====================================================
# ERROR CALCULATION
# =====================================================
test_predictions = model.predict(X_test, verbose=0)
test_predictions = scaler.inverse_transform(test_predictions)

last_log_value = data_log.iloc[lookback + split - 1].values[0]
reconstructed = []

current_value = last_log_value
for diff in test_predictions:
    current_value += diff[0]
    reconstructed.append(current_value)

reconstructed = np.expm1(reconstructed)

actual_values = df[target_col].iloc[
    lookback + split:
].values[:len(reconstructed)]

mape = mean_absolute_percentage_error(actual_values, reconstructed) * 100

# =====================================================
# METRICS
# =====================================================
max_value = forecast_df["Forecast"].max()
max_date = forecast_df.loc[
    forecast_df["Forecast"].idxmax(), "Date"
]

growth_percent = (
    (forecast_df["Forecast"].iloc[-1] - df[target_col].iloc[-1])
    / df[target_col].iloc[-1]
) * 100

# =====================================================
# PLOT
# =====================================================
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index,
    y=df[target_col],
    mode="lines",
    name="Actual (Monthly)"
))

fig.add_trace(go.Scatter(
    x=forecast_df["Date"],
    y=forecast_df["Forecast"],
    mode="lines+markers",
    name="Forecast"
))

fig.update_layout(
    title=f"{projection_mode}",
    template="plotly_white",
    height=550
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# DISPLAY METRICS
# =====================================================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Max Projected Value", f"{max_value:,.2f}")
col2.metric("Date of Maximum", max_date.strftime("%Y-%m-%d"))
col3.metric("Model Error (MAPE %)", f"{mape:.2f}%")
col4.metric("Growth Over Range (%)", f"{growth_percent:.2f}%")

st.markdown("### Forecast Table")
st.dataframe(forecast_df, use_container_width=True)

st.success("Projection generated successfully.")
