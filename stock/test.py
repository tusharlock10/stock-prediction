import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


def test_model(ticker, ):
    # load the model from the models directory
    model = load_model(f'{dir_path}/models/{ticker}_model.h5')

    # Training data is from 2010 to 30 days from today
    start = dt.datetime(2010, 1, 1)
    end = dt.datetime.today() - dt.timedelta(days=30)

    # Test data is 30 days from today to today
    test_start = dt.datetime.today() - dt.timedelta(days=30)
    test_end = dt.datetime.today()

    data = web.DataReader(ticker, 'stooq', start, end)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    prediction_days = 7  # How many days to look in the past to base our prediction out of

    test_data = web.DataReader(ticker, 'stooq', test_start, test_end)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    # Prepare the inputs

    model_inputs = total_dataset[len(total_dataset) - len(test_data) -
                                 prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get predictions

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Plot the predictions

    plt.plot(actual_prices, color="blue", label="Actual")
    plt.plot(predicted_prices, color="green", label="Predicted")
    plt.show()