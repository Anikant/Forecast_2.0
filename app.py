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
# REPRODUCIBILITY
# ----------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

st.title("📊 UPI Transaction Forecasting (Monthly + Daily LSTM)")

# ----------------------------
# FILE UPLOADER (FIXES YOUR ERROR)
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload merged_upi_transactions.xlsx",
    type=["xlsx"]
)

if uploaded_file is None:
    st.warning("Please upload the training Excel file to proceed.")
    st.stop()

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)

    df.columns = df.columns.str.strip()

    required_cols = [
        "DATE",
        "Total UPI financial transactional logs",
        "Total UPI non financial transactional logs",
        "Total UPI Transactions"
    ]

    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            st.stop()

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"])
    df = df.sort_values("DATE")
    df.set_index("DATE", inplace=True)

    return df


df = load_data(uploaded_file)
