import os
import time
import logging
import pandas as pd
import numpy as np
import talib
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from binance.client import Client
from bokeh.plotting import figure, show
from bokeh.models import Range1d
from bokeh.io import output_notebook

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API keys and other sensitive information should be kept in environment variables
api_key = os.environ.get('BINANCE_API_KEY', 'YttOvLnZvXCDvCUaYmqHbgrcq8SEk8hxPiAsOcS6y4hxk2aki2dnvctKzLPa0zUi')
api_secret = os.environ.get('BINANCE_API_SECRET', 'Bq1Ms14chZFI1Gk2oTHkDhnks3eQ8Y76QoyIu5yTzX8207jnkMwNfavpJTwB3FLP')
coinmarketcap_api_key = os.environ.get('COINMARKETCAP_API_KEY', 'e75fb4c4-0e8c-49e5-b18c-941df88edc0b')

# Binance client setup
client = Client(api_key, api_secret)

def get_symbol_klines(symbol, interval, limit):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def additional_data_available():
    return False

def append_additional_data(df):
    return df

def class_imbalance_detected(labels):
    return False

def apply_smote(df, labels):
    smote = SMOTE()
    features, labels = smote.fit_resample(df[['open', 'high', 'low', 'close', 'volume']], labels)
    df = pd.DataFrame(features, columns=['open', 'high', 'low', 'close', 'volume'])
    df['target'] = labels
    return df

def add_noise(df, labels):
    return df

def apply_elastic_transformations(df, labels):
    return df

def oversample_minority_class(df, labels):
    oversampler = SMOTE(sampling_strategy='minority')
    features, labels = oversampler.fit_resample(df[['open', 'high', 'low', 'close', 'volume']], labels)
    df = pd.DataFrame(features, columns=['open', 'high', 'low', 'close', 'volume'])
    df['target'] = labels
    return df

def undersample_majority_class(df, labels):
    undersampler = RandomUnderSampler(sampling_strategy='majority')
    features, labels = undersampler.fit_resample(df[['open', 'high', 'low', 'close', 'volume']], labels)
    df = pd.DataFrame(features, columns=['open', 'high', 'low', 'close', 'volume'])
    df['target'] = labels
    return df

def prepare_data(df, target_col='target', window=5, features_to_use=None):
    try:
        # Check if 'target' column already exists
        if 'target' not in df.columns:
            df['target'] = (df['close'].shift(-window) > df['close']).astype(int)
    except Exception as e:
        logger.error(f"Error in prepare_data - Target Calculation: {e}")

    try:
        if features_to_use is None:
            features_to_use = ['open', 'high', 'low', 'close', 'volume']

        features = df[features_to_use]
        labels = df[target_col]

        if additional_data_available():
            df = append_additional_data(df)

        if class_imbalance_detected(labels):
            df = apply_smote(df, labels)
            df = add_noise(df, labels)
            df = apply_elastic_transformations(df, labels)

        if class_imbalance_detected(labels):
            df = oversample_minority_class(df, labels)
            df = undersample_majority_class(df, labels)

        return features, labels, features_to_use

    except Exception as e:
        logger.error(f"Error in prepare_data - Feature and Label Extraction: {e}")

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)
    return model

def predict_and_trade(model, X):
    prediction = model.predict(X)
    return prediction

def lorentzian_classification(df):
    window_size = 20
    df['rolling_mean'] = df['close'].rolling(window=window_size).mean()

    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['rolling_mean'] = pd.to_numeric(df['rolling_mean'], errors='coerce')

    df['lorentzian_signal'] = np.where(df['close'] > df['rolling_mean'], 1, 0)
    return df['lorentzian_signal'].iloc[-1]

def squeezed_momentum_indicator(df):
    window_size = 20
    df['rolling_mean'] = df['close'].rolling(window=window_size).mean()
    df['upper_band'] = df['rolling_mean'] + 2 * df['close'].rolling(window=window_size).std()
    df['lower_band'] = df['rolling_mean'] - 2 * df['close'].rolling(window=window_size).std()
    df['squeeze_signal'] = np.where((df['close'] > df['lower_band']) & (df['close'] < df['upper_band']), 1, 0)
    df['rsi'] = talib.RSI(df['close'], timeperiod=window_size)
    df['momentum_signal'] = np.where(df['rsi'] > 70, 1, 0)
    df['squeezed_momentum_signal'] = df['squeeze_signal'] & df['momentum_signal']
    return df['squeezed_momentum_signal'].iloc[-1]

def get_coinmarketcap_data(api_key, symbol):
    url = f'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    parameters = {
        'start':'1',
        'limit':'10',
        'convert':'USD'
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': api_key,
    }
    
    try:
        response = requests.get(url, params=parameters, headers=headers)
        data = response.json()
        for coin in data['data']:
            if coin['symbol'] == symbol:
                return coin
    except Exception as e:
        logger.error(f"Error fetching CoinMarketCap data: {e}")
        return None

def place_order(symbol, side, quantity):
    try:
        order = client.create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=quantity
        )
        return order
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        return None

def get_user_input():
    symbol_to_trade = input("Enter the trading pair (e.g., BTCUSDT): ").upper()
    interval = input("Enter the time interval (e.g., 1m, 5m, 1h): ")
    window_size = int(input("Enter the window size for feature calculation: "))
    investment_amount = float(input("Enter the amount to invest in trading ($): "))
    return symbol_to_trade, interval, window_size, investment_amount

def choose_strategy(df, logistic_regression_model, scaler, symbol_to_trade, features_to_use):
    if len(df) <= 1:
        return "Insufficient data for strategy selection"

    last_data = df[-1:].drop(columns=['target'])
    
    # Calculate signals
    last_data_scaled = scaler.transform(last_data[features_to_use])
    logistic_regression_signal = predict_and_trade(logistic_regression_model, last_data_scaled)
    lorentzian_signal = lorentzian_classification(df)
    momentum_signal = squeezed_momentum_indicator(df)

    # Fetch additional data
    coinmarketcap_data = get_coinmarketcap_data(coinmarketcap_api_key, symbol_to_trade)
    if coinmarketcap_data:
        market_cap = coinmarketcap_data['quote']['USD']['market_cap']
        volume_24h = coinmarketcap_data['quote']['USD']['volume_24h']
        percent_change_24h = coinmarketcap_data['quote']['USD']['percent_change_24h']
        print(f"Market Cap: {market_cap}")
        print(f"24h Volume: {volume_24h}")
        print(f"24h Change: {percent_change_24h}%")

    # Calculate strategy confidence
    strategy_confidence = {
        "Logistic Regression": logistic_regression_signal,
        "Lorentzian Classification": lorentzian_signal,
        "Squeezed Momentum Indicator": momentum_signal
    }

    # Define thresholds for each signal
    logistic_regression_threshold = 0.5  # Adjust as needed
    lorentzian_threshold = 0.5  # Adjust as needed
    momentum_threshold = 0.5  # Adjust as needed

    # Combine signals based on thresholds
    combined_signal = (logistic_regression_signal > logistic_regression_threshold) + \
                      (lorentzian_signal > lorentzian_threshold) + \
                      (momentum_signal > momentum_threshold)

    if combined_signal >= 2:
        chosen_strategy = "Combined Strategy"
    elif logistic_regression_signal > logistic_regression_threshold:
        chosen_strategy = "Logistic Regression"
    elif lorentzian_signal > lorentzian_threshold:
        chosen_strategy = "Lorentzian Classification"
    elif momentum_signal > momentum_threshold:
        chosen_strategy = "Squeezed Momentum Indicator"
    else:
        chosen_strategy = "No Clear Signal"

    return chosen_strategy

def main():
    print("Welcome to the Trading Bot!")

    symbol_to_trade, interval, window_size, investment_amount = get_user_input()

    start_time = time.time()

    while True:
        try:
            elapsed_time = time.time() - start_time
            print(f"\nElapsed Time: {elapsed_time:.2f} seconds")

            klines = get_symbol_klines(symbol_to_trade, interval, 100)

            estimated_time = elapsed_time * 1.5
            print(f"Estimated Time for Fetching and Analyzing Data: {estimated_time:.2f} seconds")

            features, labels, features_to_use = prepare_data(klines, window=window_size)

            scaler = StandardScaler().fit(features)

            scaled_features = scaler.transform(features)
            X_train, _, y_train, _ = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)

            if len(np.unique(y_train)) > 1:
                logistic_regression_model = train_logistic_regression(X_train, y_train)
                chosen_strategy = choose_strategy(klines, logistic_regression_model, scaler, symbol_to_trade, features_to_use)

                print("\nAnalysis Results:")
                print(f"Chosen Strategy: {chosen_strategy}")
                print(f"Logistic Regression Signal: {logistic_regression_model.predict([scaled_features[-1]])[0]}")
                print(f"Lorentzian Signal: {lorentzian_classification(klines)}")
                print(f"Squeezed Momentum Signal: {squeezed_momentum_indicator(klines)}")

            print("Progress: Fetching and analyzing data. Sleeping for 60 seconds...")
            time.sleep(5)

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            print("Progress: Sleeping for 5 seconds...")
            time.sleep(60)

if __name__ == "__main__":
    main()
