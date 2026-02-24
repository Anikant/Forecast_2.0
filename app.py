import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------
# REPRODUCIBILITY (VERY IMPORTANT)
# ----------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("merged_upi_transactions.xlsx")

    # Clean column names
    df.columns = df.columns.str.strip()

    required_cols = [
        "DATE",
        "Total UPI financial transactional logs",
        "Total UPI non financial transactional logs",
        "total upi transactions"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"])
    df = df.sort_values("DATE")

    df.set_index("DATE", inplace=True)

    return df


df = load_data()

st.title("📊 UPI Transaction Forecasting (Monthly + Daily LSTM)")

# ----------------------------
# MONTHLY AGGREGATION
# ----------------------------
monthly_df = df.resample("M").sum()

# ----------------------------
# LSTM TRAINING FUNCTION
# ----------------------------
def train_lstm(data, lookback=6):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=False, input_shape=(X.shape[1], X.shape[2])),
        Dense(X.shape[2])
    ])

    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)

    model.fit(X, y, epochs=200, batch_size=8, verbose=0, callbacks=[early_stop])

    return model, scaler


# ----------------------------
# MONTHLY MODEL TRAINING
# ----------------------------
monthly_model, monthly_scaler = train_lstm(monthly_df.values, lookback=6)

# ----------------------------
# DAILY MODEL TRAINING
# ----------------------------
daily_model, daily_scaler = train_lstm(df.values, lookback=30)

# ----------------------------
# DAILY PROJECTION BUTTON
# ----------------------------
st.subheader("🔵 Daily Projection")

if st.button("Generate Daily Projection"):

    forecast_days = st.slider("Select number of days to forecast", 7, 180, 30)

    data = df.values
    lookback = 30

    scaled_data = daily_scaler.transform(data)
    last_sequence = scaled_data[-lookback:]

    predictions = []

    current_sequence = last_sequence.copy()

    for _ in range(forecast_days):
        pred = daily_model.predict(current_sequence.reshape(1, lookback, data.shape[1]), verbose=0)
        predictions.append(pred[0])
        current_sequence = np.vstack((current_sequence[1:], pred))

    predictions = daily_scaler.inverse_transform(predictions)

    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D"
    )

    forecast_df = pd.DataFrame(
        predictions,
        columns=df.columns,
        index=future_dates
    )

    st.write("### 📈 Daily Forecast")
    st.dataframe(forecast_df)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["total upi transactions"], label="Historical")
    ax.plot(forecast_df.index, forecast_df["total upi transactions"], label="Forecast")
    ax.legend()
    st.pyplot(fig)


# ----------------------------
# MONTHLY PROJECTION
# ----------------------------
st.subheader("🟢 Monthly Projection")

forecast_months = st.slider("Select number of months to forecast", 3, 24, 6)

data = monthly_df.values
lookback = 6

scaled_data = monthly_scaler.transform(data)
last_sequence = scaled_data[-lookback:]

predictions = []
current_sequence = last_sequence.copy()

for _ in range(forecast_months):
    pred = monthly_model.predict(current_sequence.reshape(1, lookback, data.shape[1]), verbose=0)
    predictions.append(pred[0])
    current_sequence = np.vstack((current_sequence[1:], pred))

predictions = monthly_scaler.inverse_transform(predictions)

future_dates = pd.date_range(
    start=monthly_df.index[-1] + pd.offsets.MonthEnd(),
    periods=forecast_months,
    freq="M"
)

forecast_df = pd.DataFrame(
    predictions,
    columns=monthly_df.columns,
    index=future_dates
)

st.write("### 📈 Monthly Forecast")
st.dataframe(forecast_df)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(monthly_df.index, monthly_df["total upi transactions"], label="Historical")
ax.plot(forecast_df.index, forecast_df["total upi transactions"], label="Forecast")
ax.legend()
st.pyplot(fig)
