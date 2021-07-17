import datetime as dt
from os import environ

from flask import Flask, jsonify, request
from flask_cors import CORS

from stock import predict

app = Flask(__name__)
CORS(app)

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


def check_ticker(ticker):
    if ticker not in tickers.values():
        return {'success': False, 'error': 'Ticker not recognized'}
    else:
        return {'success': True, 'error': ''}


def check_date_string(date_string):
    # Check if the date string is between 2010-01-01 and today
    result = {'success': True, 'error': ''}
    try:
        min_date = dt.datetime(2010, 1, 1)
        date = dt.datetime.strptime(date_string, '%Y-%m-%d')
        max_date = dt.datetime.today() + dt.timedelta(days=7)
        if not (min_date <= date <= max_date):
            result['success'] = False
            result['error'] = f"Date should be from {min_date.strptime('%Y-%m-%d')} to {max_date.strptime('%Y-%m-%d')}"
    except:
        result['success'] = False
        result['error'] = 'Date in invalid format, should be YYYY-MM-DD'
    return result


@app.route('/prediction', methods=['GET'])
def _prediction():
    result = None
    date_string = request.args.get('date')
    ticker = request.args.get('ticker')

    validation1 = check_date_string(date_string)
    validation2 = check_ticker(ticker)
    validation = {'success': False, 'errors': []}

    if (validation1['success'] and validation2['success']):
        result = predict.predict(ticker, date_string)
    else:
        if not validation1['success']:
            validation['errors'].append(validation1['error'])
        if not validation2['success']:
            validation['errors'].append(validation2['error'])
        result = validation

    return jsonify(result)


@app.route('/tickers', methods=['GET'])
def _tickers():
    return jsonify(tickers)


@app.route('/', methods=['GET'])
def _home():
    return jsonify("Stock Prediction API")


app.run(debug=False, host='0.0.0.0', port=environ.get("PORT", 5000))
