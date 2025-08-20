import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import os


# Load model and scaler
model = load_model('goog_lstm.keras')
scaler = joblib.load('goog_scaler.pkl')

# Streamlit UI
st.title('Stock Price Predictor')
stock = st.text_input('Enter Stock Symbol (e.g., GOOG, AAPL):', 'GOOG')
n_future = st.slider('Days to Predict:', 1, 60, 30)

# Download data
end_date = datetime.now().strftime('%Y-%m-%d')
data = yf.download(stock, start='2020-01-01', end=end_date)

if data.empty:
    st.error("No data found for this stock symbol!")
else:
    # Prepare last 100 days for prediction
    last_100_days = data['Close'].values[-100:].reshape(-1, 1)
    last_100_days_scaled = scaler.transform(last_100_days)  # <-- 2D input here

    # Predict future prices
    predictions = []
    for _ in range(n_future):
        # Reshape to 3D for LSTM *after* scaling
        x = last_100_days_scaled[-100:].reshape(1, 100, 1)  # (samples=1, timesteps=100, features=1)
        pred = model.predict(x, verbose=0)
        predictions.append(pred[0][0])
        # Update input for next prediction (2D -> scale -> 3D)
        new_input = np.append(last_100_days_scaled, pred).reshape(-1, 1)
        last_100_days_scaled = new_input[-100:]  # Keep last 100 points

    # Inverse transform predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    # Generate future dates
    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, n_future + 1)]
    
    # Plot
    st.subheader(f'Predicted Prices for Next {n_future} Days')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Historical Price', color='blue')
    ax.plot(future_dates, predictions, 'r-', label='Predicted Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid()
    st.pyplot(fig)
    
    st.subheader('Price vs MA50 vs MA100')
    ma_100_days = data['Close'].rolling(100).mean()
    fig2 = plt.figure(figsize=(8, 6))
    plt.plot(ma_100_days, 'r', label='MA100')
    plt.plot(data.Close, 'g', label='Price')
    plt.plot(ma_100_days, 'b', label='MA50') 
    plt.legend()
    st.pyplot(fig2)