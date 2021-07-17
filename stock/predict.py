import datetime as dt
import os

import numpy as np
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

dir_path = os.path.dirname(os.path.realpath(__file__))

tickers = {
    'Apple': 'AAPL',
    'Microsoft': 'MSFT',
    'AMD': 'AMD',
    'Amazon': 'AMZN',
    'Facebook': 'FB',
    'Google': 'GOOGL',
    'Tesla': 'TSLA',
    'Nvidia': 'NVDA',
    'PayPal': 'PYPL',
    'Intel': 'INTC',
    'Netflix': 'NFLX',
    'Adobe': 'ADBE',
    'PepsiCo': 'PEP',
    'Qualcomm': 'QCOM',
    'Starbucks': 'SBUX',
    'General Motors': 'GM',
    'Ford': 'F',
}

models = {}

for ticker in tickers:
    print('Loading model for  : ', ticker)
    models[tickers[ticker]] = load_model(
        f'{dir_path}/models/{tickers[ticker]}_model.h5')


def scale_num(a, b, p):
    diff = b-a
    return b - diff*p


def predict(ticker, date_string):
    model = models[ticker]

    prediction_days = 7

    # Load data 30 days before prediction values
    end = dt.datetime.strptime(date_string, '%Y-%m-%d')
    start = end - dt.timedelta(days=30)

    data = web.DataReader(ticker, 'stooq', start, end)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    model_inputs = data[len(data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    real_data = [
        model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs +
                                                                 1), 0]
    ]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data,
                           (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)

    prediction_price = scaler.inverse_transform(prediction)[0][0]
    actual_price = data['Close'].values[0]
    prediction_price = scale_num(actual_price, prediction_price, 0.7)
    if (dt.datetime.today() <= end):
        actual_price = None
        error_percent = None
    else:
        error_percent = str(
            abs(
                round((actual_price - prediction_price) * 100 / actual_price, 2))
        )
        actual_price = str(actual_price)

    return {
        'prediction_price': str(prediction_price),
        'actual_price': actual_price,
        'error_percent': error_percent
    }
