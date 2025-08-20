# 🚀 Stock Price Predictor - LSTM Neural Network Edition

A slick, Streamlit application that harnesses the power of Long Short-Term Memory (LSTM) neural networks to predict future stock prices. Pulls real-time data, makes predictions, and looks damn good doing it.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=Keras&logoColor=white)
![YFinance](https://img.shields.io/badge/YFinance-00AB06?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=NumPy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=Pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=Python&logoColor=white)

## ✨ Features

- **Real-time Data Fetching**: Pulls live stock data directly from Yahoo Finance API
- **AI-Powered Predictions**: Utilizes pre-trained LSTM model for accurate forecasting
- **Interactive UI**: Clean Streamlit interface with intuitive controls
- **Customizable Forecast**: Predict 1-60 days into the future for any stock symbol
- **Visual Analytics**: Beautiful matplotlib charts with historical data, predictions, and moving averages
- **Model Persistence**: Pre-trained model and scaler for immediate use

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/stock-price-predictor.git
   cd stock-price-predictor

2. Install dependencies
pip install -r requirements.txt

3. Train the model (first time setup)
python train_model.py

4.Run the application
streamlit run app.py


🎮 How to Use

    Enter Stock Symbol: Type any valid stock ticker (e.g., AAPL, MSFT, TSLA, GOOGL)

    Adjust Prediction Period: Use the slider to select how many days to predict (1-60)

    View Results: Watch as the app generates:

        Historical price chart with future predictions

        Moving averages (MA50/MA100) technical analysis

        Clean, professional visualizations



        Important Disclaimer

🚨 READ THIS BEFORE USING FOR TRADING 🚨

This application is for EDUCATIONAL AND DEMONSTRATION PURPOSES ONLY.

    ❌ NOT financial advice

    ❌ NOT a guaranteed prediction system

    ❌ NOT responsible for trading losses

    ✅ IS a cool tech demo

    ✅ IS educational material

    ✅ IS open source software

The stock market is volatile and unpredictable. Past performance does not indicate future results. 



Contributions are Welcome.
