import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
# Write a Python program to predict Tesla stock price using the LSTM algorithm.
#  Predict the rate of change in Tesla's stock price from tomorrow to a week from tomorrow and draw a graph, reflecting stock price fluctuations, economic conditions, interest rates, etc. that cause the fluctuations.
# And, output it with the accuracy of the algorithm.


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
    print(predicted_price_scaled)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]
    result = f"Predicted {arg} Stock Price for To day: {stock_data.iloc[-1]['Close']:.2f}, Tomorrow: {predicted_price:.2f}"
    print(result)

    # Plot the historical Tesla stock price
    # plt.figure(figsize=(12, 6))
    # plt.plot(data.index, data['Close'], label='Historical Stock Price')
    # plt.title(f"{arg} Stock Price Over 10 Years")
    # plt.xlabel("Date")
    # plt.ylabel("Stock Price")
    # plt.legend()
    # plt.show()
    return result


future_days = 7


def tesla():
    # Download Tesla stock price data from Yahoo Finance
    ticker = "AMZN"
    start_date = "2010-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')

    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Close']]

    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Define a function to create sequences of data
    def create_sequences(data, lookback):
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i+lookback])
            y.append(data[i+lookback])
        return np.array(X), np.array(y)

    lookback = 60
    X, y = create_sequences(data_scaled, lookback)

    # Split data into training and testing sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build and train an LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(lookback, 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=100)

    # Make predictions for the next week

    last_sequence = data_scaled[-lookback:]
    predicted_change = []

    for _ in range(future_days):
        next_day = model.predict(np.array([last_sequence]))
        predicted_change.append(next_day[0][0])
        last_sequence = np.append(last_sequence[1:], next_day, axis=0)

    # Inverse transform the predictions to get actual prices
    predicted_change = scaler.inverse_transform(
        np.array(predicted_change).reshape(-1, 1))

    # Plot the predicted rate of change
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, future_days + 1), predicted_change)
    plt.title(f"Predicted Rate of Change in {ticker} Stock Price ({future_days} days)")
    plt.xlabel("Days")
    plt.ylabel("Rate of Change")
    plt.grid(True)

    # Print the predicted change
    print(f"Predicted Rate of Change in {ticker} Stock Price ({{future_days} days}):")
    for i, change in enumerate(predicted_change):
        print(f"Day {i+1}: {change[0]}")

    plt.show()
