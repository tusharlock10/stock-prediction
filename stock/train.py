import datetime as dt
import time

import numpy as np
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

tickers = {
    'Qualcomm': 'QCOM',
    'Starbucks': 'SBUX',
    'General Motors': 'GM',
    'Ford': 'F',
}


def train_model(ticker):
    # Training data is from 2010 to 30 days from today
    start = dt.datetime(2010, 1, 1)
    end = dt.datetime.today() - dt.timedelta(days=30)

    # Load Data
    data = web.DataReader(ticker, 'stooq', start, end)

    # Prepare data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    prediction_days = 7  # How many days to look in the past to base our prediction out of

    # Get training data
    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the model
    model = Sequential()
    model.add(
        LSTM(units=50,
             return_sequences=True,
             input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(units=1))  # Prediction of closing value

    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train the model
    start = time.time()
    model.fit(x_train, y_train, epochs=50, batch_size=32)
    print("Training completed in : ", time.time() - start)

    model.save(f'models/{ticker}_model.h5', save_format='h5')


if (__name__ == "__main__"):
    for ticker in tickers:
        train_model(tickers[ticker])
