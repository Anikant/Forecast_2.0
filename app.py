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
# 🔒 FULL DETERMINISM (CRITICAL)
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
UPI Transaction Forecasting Engine (Advanced Deterministic LSTM)
</h1>
""", unsafe_allow_html=True)

st.markdown("---")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    file_path = "data/UPI_Transactions.xlsx"

    if not os.path.exists(file_path):
        st.error("UPI_Transactions.xlsx not found in data folder.")
        st.stop()

    df = pd.read_excel(file_path, engine="openpyxl")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)

    return df

df = load_data()

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("Forecast Settings")

fields = ["Remitter", "Benificiary", "Total"]

selected_field = st.sidebar.selectbox(
    "Select Projection Field",
    fields
)

forecast_months = st.sidebar.slider(
    "Forecast Horizon (Months)",
    min_value=1,
    max_value=24,
    value=6
)

# =====================================================
# DATA PREPARATION
# =====================================================
data = df[[selected_field]].dropna()

data_log = np.log1p(data)
data_diff = data_log.diff().dropna()

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_diff)

lookback = 18

def create_sequences(dataset, lookback):
    X, y = [], []
    for i in range(lookback, len(dataset)):
        X.append(dataset[i - lookback:i])
        y.append(dataset[i])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, lookback)

if len(X) < 10:
    st.error("Not enough historical data for training.")
    st.stop()

split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =====================================================
# MODEL BUILD (CACHED)
# =====================================================
@st.cache_resource
def train_model(X_train, y_train, X_test, y_test):

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
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
        epochs=150,
        batch_size=8,
        validation_data=(X_test, y_test),
        shuffle=False,          # 🔥 critical for determinism
        callbacks=[early_stop],
        verbose=0
    )

    return model

with st.spinner("Training deterministic LSTM model..."):
    model = train_model(X_train, y_train, X_test, y_test)

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

actual_values = df[selected_field].iloc[
    lookback + split:
].values[:len(reconstructed)]

mape = mean_absolute_percentage_error(actual_values, reconstructed) * 100

# =====================================================
# FUTURE FORECAST
# =====================================================
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

# =====================================================
# FORECAST DATAFRAME
# =====================================================
last_date = df.index[-1]

future_dates = [
    last_date + pd.DateOffset(months=i + 1)
    for i in range(forecast_months)
]

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast": future_values
})

# =====================================================
# GROWTH %
# =====================================================
start_value = df[selected_field].iloc[-1]
end_value = future_values[-1]

growth_percent = ((end_value - start_value) / start_value) * 100

# =====================================================
# MAX PROJECTION
# =====================================================
max_value = forecast_df["Forecast"].max()
max_date = forecast_df.loc[
    forecast_df["Forecast"].idxmax(), "Date"
]

# =====================================================
# PLOT
# =====================================================
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index,
    y=df[selected_field],
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
    title=f"{selected_field} Forecast Projection",
    template="plotly_white",
    height=550,
    xaxis_title="Date",
    yaxis_title="Transaction Count"
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# METRICS
# =====================================================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Max Projected Value", f"{max_value:,.2f}")
col2.metric("Date of Maximum", max_date.strftime("%Y-%m"))
col3.metric("Model Error (MAPE %)", f"{mape:.2f}%")
col4.metric("Growth Over Range (%)", f"{growth_percent:.2f}%")

st.markdown("### Forecast Table")
st.dataframe(forecast_df, use_container_width=True)

st.success("Deterministic forecast generated successfully.")
