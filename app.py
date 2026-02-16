import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import datetime

st.set_page_config(page_title="UPI LSTM Forecast", layout="wide")

# ===============================
# UI HEADER
# ===============================
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>
    UPI Transaction Forecasting System (LSTM)
    </h1>
""", unsafe_allow_html=True)

st.markdown("---")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_excel("data/UPI_Transactions.xlsx")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)
    return df

df = load_data()

fields = ["Remitter", "Benificiary", "Total"]

# ===============================
# USER INPUT
# ===============================
st.sidebar.header("Forecast Settings")

forecast_months = st.sidebar.slider(
    "Select Forecast Months",
    min_value=1,
    max_value=24,
    value=6
)

selected_field = st.sidebar.selectbox(
    "Select Field to Forecast",
    fields
)

# ===============================
# DATA PREPARATION
# ===============================
data = df[[selected_field]].dropna()

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

lookback = 12

def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, lookback)

split = int(len(X)*0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ===============================
# BUILD LSTM MODEL
# ===============================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(patience=5, restore_best_weights=True)

model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=0
)

# ===============================
# TEST PREDICTIONS
# ===============================
predictions = model.predict(X_test)

predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)

mape = mean_absolute_percentage_error(y_test_actual, predictions) * 100

# ===============================
# FUTURE FORECAST
# ===============================
last_sequence = scaled_data[-lookback:]
future_forecast = []

current_sequence = last_sequence.reshape(1, lookback, 1)

for _ in range(forecast_months):
    next_pred = model.predict(current_sequence)[0]
    future_forecast.append(next_pred)
    current_sequence = np.append(current_sequence[:,1:,:], [[next_pred]], axis=1)

future_forecast = scaler.inverse_transform(future_forecast)

last_date = df.index[-1]
future_dates = [
    last_date + pd.DateOffset(months=i+1)
    for i in range(forecast_months)
]

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast": future_forecast.flatten()
})

max_value = forecast_df["Forecast"].max()
max_date = forecast_df.loc[forecast_df["Forecast"].idxmax(), "Date"]

# ===============================
# PLOT GRAPH
# ===============================
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index,
    y=df[selected_field],
    mode='lines',
    name='Actual'
))

fig.add_trace(go.Scatter(
    x=forecast_df["Date"],
    y=forecast_df["Forecast"],
    mode='lines+markers',
    name='Forecast'
))

fig.update_layout(
    title=f"{selected_field} Forecast",
    template="plotly_white",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# ===============================
# RESULTS SECTION
# ===============================
col1, col2, col3 = st.columns(3)

col1.metric("Max Projected Value", f"{max_value:,.0f}")
col2.metric("Date of Maximum", max_date.strftime("%Y-%m"))
col3.metric("Model Error (MAPE %)", f"{mape:.2f}%")

st.markdown("### Forecast Table")
st.dataframe(forecast_df)

