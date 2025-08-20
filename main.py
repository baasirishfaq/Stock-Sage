import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

# App title and description
st.set_page_config(page_title="Stock Prediction App", layout="wide")
st.title("📈 Stock Market Prediction")
st.write("Predict future stock prices using LSTM neural networks")

# Sidebar for user inputs
with st.sidebar:
    st.header("Settings")
    stock = st.text_input("Stock Symbol", "GOOG")
    days_to_predict = st.slider("Days to Predict", 1, 30, 7)

# Load model with error handling
try:
    model = load_model('goog_lstm.keras')
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# Download data
@st.cache_data
def load_data(stock):
    return yf.download(stock, start='2015-01-01', end=pd.Timestamp.today())

try:
    data = load_data(stock)
    if data.empty:
        st.warning("No data found for this stock symbol")
        st.stop()
except Exception as e:
    st.error(f"Error downloading data: {str(e)}")
    st.stop()

# Display raw data
st.subheader(f"Historical Data for {stock}")
st.dataframe(data.tail(), use_container_width=True)

# Data processing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']])

# Prepare sequences for prediction
def prepare_data(data, lookback=100):
    X = []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
    return np.array(X)

latest_sequence = scaled_data[-100:]
future_predictions = []

# Make predictions
for _ in range(days_to_predict):
    x = latest_sequence.reshape(1, 100, 1)
    pred = model.predict(x, verbose=0)[0][0]
    future_predictions.append(pred)
    latest_sequence = np.append(latest_sequence[1:], [[pred]], axis=0)

# Inverse transform predictions
future_predictions = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1, 1)
)

# Create prediction dataframe
future_dates = pd.date_range(
    start=data.index[-1] + pd.Timedelta(days=1),
    periods=days_to_predict
)
predictions_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Price': future_predictions.flatten()
})

# Visualization
st.subheader(f"Next {days_to_predict} Days Prediction")
st.line_chart(predictions_df.set_index('Date'))

# Show predictions table
st.write("Predicted Prices:")
st.dataframe(predictions_df, use_container_width=True)

# Add download button
st.download_button(
    label="Download Predictions",
    data=predictions_df.to_csv(index=False),
    file_name=f"{stock}_predictions.csv",
    mime="text/csv"
)