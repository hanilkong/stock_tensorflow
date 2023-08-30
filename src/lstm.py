import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt


def stock(arg):

    # Download Tesla stock data from Yahoo Finance
    stock_data = yf.download(arg, start='2012-01-01',
                             end=datetime.today().strftime('%Y-%m-%d'))

    # Extract Close prices
    data = stock_data[['Close']]

    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Prepare data for LSTM
    look_back = 60  # Number of previous days' data to consider
    X, y = [], []
    for i in range(len(data_scaled) - look_back):
        X.append(data_scaled[i:i+look_back])
        y.append(data_scaled[i+look_back])
    X, y = np.array(X), np.array(y)

    # Split data into train and test sets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(
            X_train.shape[1], X_train.shape[2])),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Predict tomorrow's price
    last_60_days = data_scaled[-look_back:]
    last_60_days = np.expand_dims(last_60_days, axis=0)
    predicted_price_scaled = model.predict(last_60_days)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]
    result = f"Predicted {arg} Stock Price for To day: {stock_data.iloc[-1]['Close']:.2f}, Tomorrow: {predicted_price:.2f}"
    print(result)

    # Plot the historical Tesla stock price
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Historical Stock Price')
    plt.title(f"{arg} Stock Price Over 10 Years")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()
    return result
