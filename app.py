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
# LOAD DATA (Cloud Safe)
# =====================================================
uploaded_file = st.file_uploader(
    "Upload merged_upi_transactions.xlsx",
    type=["xlsx"]
)

if uploaded_file is None:
    st.warning("Please upload merged_upi_transactions.xlsx to proceed.")
    st.stop()

@st.cache_data
def load_data(file):
    df = pd.read_excel(file)

    # Normalize column names (CRITICAL FIX)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace("  ", " ")
    )

    # Create mapping dictionary (flexible matching)
    column_mapping = {}

    for col in df.columns:
        if "date" in col:
            column_mapping[col] = "DATE"

        elif "financial" in col:
            column_mapping[col] = "Total UPI financial transactional logs"

        elif "non" in col:
            column_mapping[col] = "Total UPI non financial transactional logs"

        elif "total" in col and "upi" in col:
            column_mapping[col] = "total upi transactions"

    df = df.rename(columns=column_mapping)

    required_cols = [
        "DATE",
        "Total UPI financial transactional logs",
        "Total UPI non financial transactional logs",
        "total upi transactions"
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(f"Missing required columns: {missing}")
        st.write("Available columns detected:", list(df.columns))
        st.stop()

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"])
    df = df.sort_values("DATE")
    df.set_index("DATE", inplace=True)

    return df

df = load_data(uploaded_file)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("Forecast Settings")

fields = [
    "Total UPI FinancialTRANSACTION LOG",
    "Total UPI Non-FinancialTRANSACTION LOG",
    "total Upi Transaction"
]

selected_field = st.sidebar.selectbox(
    "Select Projection Field",
    fields
)

# 🔥 NEW DAILY PROJECTION TOGGLE (ABOVE SLIDER)
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
# DATA PREPARATION FUNCTION
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

# =====================================================
# MONTHLY MODEL
# =====================================================
if not daily_projection:

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

# =====================================================
# DAILY MODEL (SEPARATE LSTM)
# =====================================================
else:

    series = df[selected_field]

    lookback = 30
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
        series.index[-1] + pd.DateOffset(days=i+1)
        for i in range(forecast_horizon)
    ]

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
