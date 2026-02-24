import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import os

st.set_page_config(layout="wide")

# ==========================
# CONFIG
# ==========================
LOOKBACK = 60
EPOCHS = 120
BATCH_SIZE = 32

# ==========================
# LOAD DATA FUNCTION
# ==========================
@st.cache_data
def load_data(file_path):
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df

# ==========================
# PREPARE MULTIVARIATE DATA
# ==========================
def prepare_data(df, target_col):

    # keep only numeric columns
    df_numeric = df.select_dtypes(include=[np.number]).copy()

    if target_col not in df_numeric.columns:
        raise ValueError(f"{target_col} not numeric or missing.")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_numeric)

    X, y = [], []
    for i in range(LOOKBACK, len(scaled_data)):
        X.append(scaled_data[i-LOOKBACK:i])
        y.append(scaled_data[i, df_numeric.columns.get_loc(target_col)])

    return np.array(X), np.array(y), scaler, df_numeric

# ==========================
# BUILD PURE ML MODEL
# ==========================
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.Huber()
    )
    return model

# ==========================
# FORECAST FUNCTION
# ==========================
def forecast_future(model, df_numeric, scaler, target_col, steps):

    scaled_data = scaler.transform(df_numeric)
    input_seq = scaled_data[-LOOKBACK:].copy()
    predictions = []

    for _ in range(steps):
        pred = model.predict(input_seq.reshape(1, LOOKBACK, -1), verbose=0)[0][0]
        predictions.append(pred)

        next_row = input_seq[-1].copy()
        next_row[df_numeric.columns.get_loc(target_col)] = pred

        input_seq = np.vstack([input_seq[1:], next_row])

    # inverse scaling
    dummy = np.zeros((len(predictions), df_numeric.shape[1]))
    dummy[:, df_numeric.columns.get_loc(target_col)] = predictions
    inv = scaler.inverse_transform(dummy)

    return inv[:, df_numeric.columns.get_loc(target_col)]

# ==========================
# TITLE
# ==========================
st.title("UPI Pure ML Forecast Engine 2.0")

# ==========================
# MODE TOGGLE
# ==========================
daily_mode = st.toggle("Enable Daily Projection Mode")

if daily_mode:
    file_path = "data/merged_upi_transactions.xlsx"
else:
    file_path = "data/UPI_Transactions.xlsx"

if not os.path.exists(file_path):
    st.error(f"{file_path} not found in data folder.")
    st.stop()

df = load_data(file_path)

# ==========================
# DATE HANDLING
# ==========================
if "DATE" in df.columns:
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE")
    df.set_index("DATE", inplace=True)
else:
    st.error("DATE column missing.")
    st.stop()

# ==========================
# TARGET COLUMN
# ==========================
target_col = st.selectbox("Select Target Column", df.select_dtypes(include=[np.number]).columns)

# ==========================
# SLIDERS
# ==========================
if daily_mode:
    days = st.slider("Select Number of Days for Projection", 7, 90, 30)
else:
    months = st.slider("Select Number of Months for Projection", 1, 12, 3)

# ==========================
# PREPARE DATA
# ==========================
try:
    X, y, scaler, df_numeric = prepare_data(df, target_col)
except Exception as e:
    st.error(str(e))
    st.stop()

if len(X) < 100:
    st.error("Not enough data for ML training.")
    st.stop()

# ==========================
# TRAIN MODEL
# ==========================
with st.spinner("Training Pure Multivariate ML Model..."):

    model = build_model((X.shape[1], X.shape[2]))

    early_stop = EarlyStopping(
        monitor="loss",
        patience=15,
        restore_best_weights=True
    )

    model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[early_stop]
    )

# ==========================
# FORECAST
# ==========================
if daily_mode:
    future_values = forecast_future(model, df_numeric, scaler, target_col, days)
    future_index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days, freq="D")
else:
    future_values = forecast_future(model, df_numeric, scaler, target_col, months)
    future_index = pd.date_range(df.index[-1] + pd.offsets.MonthBegin(1), periods=months, freq="MS")

# ==========================
# GRAPH WINDOW (LAST 30 DAYS + FUTURE)
# ==========================
recent_data = df[target_col].last("30D")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=recent_data.index,
    y=recent_data.values,
    mode='lines',
    name="Last 30 Days"
))

fig.add_trace(go.Scatter(
    x=future_index,
    y=future_values,
    mode='lines',
    name="Projection"
))

st.plotly_chart(fig, use_container_width=True)

# ==========================
# MAX DAY LABEL (DAILY MODE)
# ==========================
if daily_mode:
    max_idx = np.argmax(future_values)
    max_day = future_index[max_idx]
    max_val = future_values[max_idx]

    st.success(f"Maximum Projected Day: {max_day.date()} | Value: {round(max_val,2)}")

st.success("Pure ML Forecast Complete.")
