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
# 🔒 FULL DETERMINISM
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
st.set_page_config(page_title="UPI LSTM Forecast", layout="wide")

st.markdown("""
<h1 style='text-align: center; color: #1F4E79;'>
UPI Transaction Forecasting Engine (Monthly + Daily Deterministic LSTM)
</h1>
""", unsafe_allow_html=True)

st.markdown("---")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("Forecast Settings")

daily_projection = st.sidebar.checkbox(
    "Enable Daily Projection (Separate LSTM Model)"
)

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon",
    min_value=1,
    max_value=30 if daily_projection else 24,
    value=7 if daily_projection else 6
)

# =====================================================
# LOAD MONTHLY DATA (DEFAULT)
# =====================================================
@st.cache_data
def load_monthly_data():
    file_path = "data/UPI_Transactions.xlsx"
    if not os.path.exists(file_path):
        st.error("UPI_Transactions.xlsx not found in data folder.")
        st.stop()

    df = pd.read_excel(file_path, engine="openpyxl")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)
    return df

# =====================================================
# LOAD DAILY DATA (ONLY WHEN ENABLED)
# =====================================================
@st.cache_data
def load_daily_data():
    file_path = "data/merged_upi_transactions.xlsx"
    if not os.path.exists(file_path):
        st.error("merged_upi_transactions.xlsx not found in data folder.")
        st.stop()

    df = pd.read_excel(file_path, engine="openpyxl")

    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE")
    df.set_index("DATE", inplace=True)

    return df

# =====================================================
# DATA PREP FUNCTION
# =====================================================
def prepare_data(series, lookback):
    data_log = np.log1p(series)
    data_diff = data_log.diff().dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data_diff.values.reshape(-1, 1))

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i])
        y.append(scaled[i])

    return np.array(X), np.array(y), scaler, data_log

# =====================================================
# MODEL TRAIN FUNCTION
# =====================================================
@st.cache_resource
def train_model(X_train, y_train, X_test, y_test, lookback):

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(patience=5, restore_best_weights=True)

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

# =====================================================
# MONTHLY MODE (DEFAULT)
# =====================================================
if not daily_projection:

    df = load_monthly_data()

    fields = ["Remitter", "Benificiary", "Total"]

    selected_field = st.sidebar.selectbox(
        "Select Projection Field",
        fields
    )

    series = df[selected_field].resample("M").sum()

    lookback = 18
    X, y, scaler, data_log = prepare_data(series, lookback)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = train_model(X_train, y_train, X_test, y_test, lookback)

    last_sequence = scaler.transform(
        data_log.diff().dropna().values.reshape(-1, 1)
    )[-lookback:]

    current_seq = last_sequence.reshape(1, lookback, 1)
    forecasts = []

    for _ in range(forecast_horizon):
        pred = model.predict(current_seq, verbose=0)[0]
        forecasts.append(pred)
        current_seq = np.append(current_seq[:, 1:, :], [[pred]], axis=1)

    forecasts = scaler.inverse_transform(forecasts)

    last_log = data_log.iloc[-1]
    future_vals = []
    curr = last_log

    for diff in forecasts:
        curr += diff[0]
        future_vals.append(curr)

    future_vals = np.expm1(future_vals)

    future_dates = [
        series.index[-1] + pd.DateOffset(months=i+1)
        for i in range(forecast_horizon)
    ]

    st.markdown("## Monthly Projection Results")

# =====================================================
# =====================================================
# DAILY MODE (SEPARATE WINDOW)
# =====================================================
else:

    df = load_daily_data()

    # Normalize columns safely
    df.columns = df.columns.str.strip()

    # Exclude DATE column automatically
    available_fields = [col for col in df.columns if col.upper() != "DATE"]

    if not available_fields:
        st.error("No valid transaction columns found in merged_upi_transactions.xlsx")
        st.stop()

    selected_field = st.sidebar.selectbox(
        "Select Daily Projection Field",
        available_fields
    )

    series = df[selected_field]

    lookback = 30
    X, y, scaler, data_log = prepare_data(series, lookback)

    if len(X) < 10:
        st.error("Not enough daily historical data for training.")
        st.stop()

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = train_model(X_train, y_train, X_test, y_test, lookback)

    last_sequence = scaler.transform(
        data_log.diff().dropna().values.reshape(-1, 1)
    )[-lookback:]

    current_seq = last_sequence.reshape(1, lookback, 1)
    forecasts = []

    for _ in range(forecast_horizon):
        pred = model.predict(current_seq, verbose=0)[0]
        forecasts.append(pred)
        current_seq = np.append(current_seq[:, 1:, :], [[pred]], axis=1)

    forecasts = scaler.inverse_transform(forecasts)

    last_log = data_log.iloc[-1]
    future_vals = []
    curr = last_log

    for diff in forecasts:
        curr += diff[0]
        future_vals.append(curr)

    future_vals = np.expm1(future_vals)

    future_dates = [
        series.index[-1] + pd.DateOffset(days=i+1)
        for i in range(forecast_horizon)
    ]

    st.markdown("## 🔥 Daily Projection Results")
# =====================================================
# FORECAST DF
# =====================================================
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast": future_vals
})

# =====================================================
# PLOT
# =====================================================
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=series.index,
    y=series.values,
    mode="lines",
    name="Actual"
))

fig.add_trace(go.Scatter(
    x=forecast_df["Date"],
    y=forecast_df["Forecast"],
    mode="lines+markers",
    name="Forecast"
))

fig.update_layout(
    template="plotly_white",
    height=550,
    xaxis_title="Date",
    yaxis_title="Transaction Count"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("### Forecast Table")
st.dataframe(forecast_df, use_container_width=True)

st.success("Deterministic forecast generated successfully.")
