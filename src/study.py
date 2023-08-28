import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def stock_study(file):
    # Load stock data (You need to have a CSV file with historical stock data)
    stock_data = pd.read_csv(file)

    # Use 'Close' prices as the target variable to predict
    y = stock_data['Close'].values

    # Feature scaling
    scaler = MinMaxScaler()
    y = y.reshape(-1, 1)
    y = scaler.fit_transform(y)

    # Split data into training and testing sets
    train_size = int(0.8 * len(y))
    train_data = y[:train_size]
    test_data = y[train_size:]

    # Create sequences for training

    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i+seq_length])
        return np.array(sequences)

    seq_length = 10
    X_train = create_sequences(train_data, seq_length)
    X_test = create_sequences(test_data, seq_length)

    # Build the neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu',
                             input_shape=(seq_length, 1)),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_train, train_data[seq_length:], epochs=50, batch_size=32)

    # Make predictions
    predicted_data = model.predict(X_test)
    predicted_data = scaler.inverse_transform(predicted_data)

    # Visualize the results
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data['Close'].values[train_size +
             seq_length:], label='Actual Prices')
    plt.plot(predicted_data, label='Predicted Prices', linestyle='dashed')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction using LSTM')
    plt.legend()
    plt.show()
