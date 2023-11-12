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
server_time = client.get_server_time()

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

def get_current_price(symbol):
    # Replace with your implementation to fetch the current price for the given symbol
    # Example for Binance API:
    ticker = client.get_symbol_ticker(symbol=symbol)
    current_price = float(ticker['price'])
    return current_price

def format_quantity(quantity, symbol_info):
    # Adjust the quantity to match the asset's allowed precision
    step_size = float([f['stepSize'] for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'][0])
    precision = int(round(-math.log(step_size, 10), 0))
    return "{:0.0{}f}".format(quantity, precision)

def place_buy_order(symbol, quantity):
    try:
        symbol_info = client.get_symbol_info(symbol)
        formatted_quantity = format_quantity(quantity, symbol_info)

        order = client.create_order(
            symbol=symbol,
            side=Client.SIDE_BUY,
            type=Client.ORDER_TYPE_MARKET,
            quantity=formatted_quantity,
            timestamp=server_time['serverTime']
        )
        return order
    except Exception as e:
        logger.error(f"Error placing buy order: {e}")
        return None

def place_sell_order(symbol, quantity):
    try:
        symbol_info = client.get_symbol_info(symbol)
        formatted_quantity = format_quantity(quantity, symbol_info)

        order = client.create_order(
            symbol=symbol,
            side=Client.SIDE_SELL,
            type=Client.ORDER_TYPE_MARKET,
            quantity=formatted_quantity,
            timestamp=server_time['serverTime']
        )
        return order
    except Exception as e:
        logger.error(f"Error placing sell order: {e}")
        return None

def place_order(symbol, side, quantity):
    try:
        order = client.create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=formatted_quantity,
            timestamp=server_time['serverTime']  # Pass the server time as the timestamp
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

def choose_strategy(df, logistic_regression_model, scaler, symbol_to_trade, features_to_use, market_data):
    if len(df) <= 1:
        return "Insufficient data for strategy selection", "Hold"

    last_data = df[-1:].drop(columns=['target'])
    last_data_scaled = scaler.transform(last_data[features_to_use])
    logistic_regression_signal = predict_and_trade(logistic_regression_model, last_data_scaled)
    lorentzian_signal = lorentzian_classification(df)
    momentum_signal = squeezed_momentum_indicator(df)

    # Market data considerations
    high_volatility = abs(market_data.get('percent_change_24h', 0)) > 5

    # Strategy decision based on market data
    if high_volatility:
        if market_data['percent_change_24h'] > 0:
            chosen_strategy = "High Volatility - Positive Change"
            action = "Sell" if logistic_regression_signal > 0.5 else "Hold"
        else:
            chosen_strategy = "High Volatility - Negative Change"
            action = "Buy" if logistic_regression_signal < 0.5 else "Hold"
    else:
        # Default strategy based on logistic regression and other indicators
        chosen_strategy, action = default_strategy_decision(logistic_regression_signal, lorentzian_signal, momentum_signal)

    return chosen_strategy, action

def default_strategy_decision(logistic_regression_signal, lorentzian_signal, momentum_signal):
    # Define thresholds for each signal
    logistic_regression_threshold = 0.5  # Adjust as needed
    lorentzian_threshold = 0.5  # Adjust as needed
    momentum_threshold = 0.5  # Adjust as needed

    # Combine signals based on thresholds
    combined_signal = (logistic_regression_signal > logistic_regression_threshold) + \
                    (lorentzian_signal > lorentzian_threshold) + \
                    (momentum_signal > momentum_threshold)

    # Determine strategy and corresponding action (Buy or Sell)
    action = "Hold"
    if logistic_regression_signal > logistic_regression_threshold:
        chosen_strategy = "Logistic Regression"
        action = "Buy"
    elif lorentzian_signal > lorentzian_threshold:
        chosen_strategy = "Lorentzian Classification"
        action = "Buy"
    elif momentum_signal > momentum_threshold:
        chosen_strategy = "Squeezed Momentum Indicator"
        action = "Buy"
    else:
        chosen_strategy = "No Clear Signal"

    return chosen_strategy, action


def main():
    print("Welcome to the Trading Bot!")

    symbol_to_trade, interval, window_size, investment_amount = get_user_input()

    executed_order = False  # Track whether a buy/sell order has been executed
    
    coinmarketcap_data = get_coinmarketcap_data(coinmarketcap_api_key, symbol_to_trade)
    market_data = {}
    if coinmarketcap_data:
        # Extract relevant metrics
        market_data['volume_24h'] = coinmarketcap_data.get('quote', {}).get('USD', {}).get('volume_24h', 0)
        market_data['percent_change_24h'] = coinmarketcap_data.get('quote', {}).get('USD', {}).get('percent_change_24h', 0)
        market_data['market_cap'] = coinmarketcap_data.get('quote', {}).get('USD', {}).get('market_cap', 0)
        print(f"CoinMarketCap Data for {symbol_to_trade}: {market_data}")
        
    while not executed_order:
                        
        try:
            start_time = time.time()
            klines = get_symbol_klines(symbol_to_trade, interval, 100)
            features, labels, features_to_use = prepare_data(klines, window=window_size)
            scaler = StandardScaler().fit(features)
            scaled_features = scaler.transform(features)
            X_train, _, y_train, _ = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)

            if len(np.unique(y_train)) > 1:
                logistic_regression_model = train_logistic_regression(X_train, y_train)
                chosen_strategy, action = choose_strategy(klines, logistic_regression_model, scaler, symbol_to_trade, features_to_use, market_data)

                print("\nAnalysis Results:")
                print(f"Chosen Strategy: {chosen_strategy}")
                print(f"Action: {action}")
                
                # Place buy or sell orders based on the action
                current_price = get_current_price(symbol_to_trade)
                if action == "Buy":
                    quantity_to_buy = investment_amount / current_price
                    place_buy_order(symbol_to_trade, quantity_to_buy)
                    print(f"Buy order placed for {quantity_to_buy} {symbol_to_trade} at {current_price}")
                    executed_order = True
                elif action == "Sell":
                    quantity_to_sell = investment_amount / current_price
                    place_sell_order(symbol_to_trade, quantity_to_sell)
                    print(f"Sell order placed for {quantity_to_sell} {symbol_to_trade} at {current_price}")
                    executed_order = True
            
            elapsed_time = time.time() - start_time
            print(f"\nElapsed Time: {elapsed_time:.2f} seconds")
        
            print("Progress: Fetching and analyzing data. Sleeping for a minute ...")
            time.sleep(60)

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            print("Progress: Sleeping for a minute ...")
            time.sleep(60)

if __name__ == "__main__":
    main()
