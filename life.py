import os
import time
import logging
import math
import pandas as pd
import numpy as np
import talib
import requests
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from binance.client import Client


# Constants
COLORS = {
    'green': '\033[92m',
    'blue': '\033[94m',
    'yellow': '\033[93m',
    'red': '\033[91m',
    'purple': '\033[95m',
    'reset': '\033[0m'
}
KNOWN_QUOTE_ASSETS = ["USDT", "BTC", "ETH", "BNB", "XRP", "LTC", "ADA", "DOT", "BCH", "LINK"]

# Custom Logging Handler
class CustomLoggingHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__()
        self.RED = '\033[91m'
        self.RESET = '\033[0m'
    def emit(self, record):
        if record.levelno == logging.ERROR:
            self.stream.write(self.RED)
        super().emit(record)
        self.stream.write(self.RESET)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = CustomLoggingHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# API keys and other sensitive information should be kept in environment variables
api_key = os.environ.get('BINANCE_API_KEY', 'YttOvLnZvXCDvCUaYmqHbgrcq8SEk8hxPiAsOcS6y4hxk2aki2dnvctKzLPa0zUi')
api_secret = os.environ.get('BINANCE_API_SECRET', 'Bq1Ms14chZFI1Gk2oTHkDhnks3eQ8Y76QoyIu5yTzX8207jnkMwNfavpJTwB3FLP')
coinmarketcap_api_key = os.environ.get('COINMARKETCAP_API_KEY', 'e75fb4c4-0e8c-49e5-b18c-941df88edc0b')

# Binance client setup
client = Client(api_key, api_secret)
server_time = client.get_server_time()


# Data Retrieval Functions

# Function to get historical klines (candlestick data) for a given symbol and interval
def get_symbol_klines(symbol, interval, limit):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# Function to get data for a specific cryptocurrency from CoinMarketCap
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

# Function to get the current mean price of a given symbol from Binance & CoinMarketCap
def get_current_price(symbol):
    # Initialize prices
    current_price_binance = None
    current_price_coinmarketcap = None

    # Fetch current price from Binance
    try:
        ticker_binance = client.get_symbol_ticker(symbol=symbol)
        current_price_binance = float(ticker_binance['price'])
    except Exception as e:
        logger.error(f"Error fetching current price from Binance: {e}")

    # Fetch current price from CoinMarketCap
    try:
        url = f'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
        parameters = {'symbol': symbol}
        headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': coinmarketcap_api_key,
        }
        response = requests.get(url, params=parameters, headers=headers)
        data = response.json()
        current_price_coinmarketcap = float(data['data'][symbol][0]['quote']['USD']['price'])
    except Exception as e:
        logger.error(f"Error fetching current price from CoinMarketCap: {e}")

    # Calculate mean price
    prices = [price for price in [current_price_binance, current_price_coinmarketcap] if price is not None]
    if prices:
        mean_price = sum(prices) / len(prices)
        return mean_price
    else:
        logger.error("Failed to fetch prices from both Binance and CoinMarketCap")
        return None

# Function to get the last 5-minute market data for a given symbol
def get_last_5min_market_data(symbol):
    try:
        klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_5MINUTE, limit=1)
        last_kline = klines[-1]  # Get the latest kline
        high_price = last_kline[2]  # High price is the 3rd element
        low_price = last_kline[3]  # Low price is the 4th element
        return high_price, low_price
    except Exception as e:
        logger.error(f"Error fetching last 5-minute market data: {e}")
        return None, None


# Utility Functions
def get_user_input():
    symbol_to_trade = input("Enter the trading pair (e.g., BTCUSDT): ").upper()
    interval = input("Enter the time interval (e.g., 1m, 5m, 1h): ")
    window_size = int(input("Enter the window size for feature calculation: "))
    investment_amount = float(input("Enter the amount to invest in trading ($): "))
    return symbol_to_trade, interval, window_size, investment_amount

# Order Execution Functions
def get_quote_asset(trading_pair):
    for quote_asset in KNOWN_QUOTE_ASSETS:
        if trading_pair.endswith(quote_asset):
            return quote_asset
    return None  # or raise an error if no known quote asset is found

def get_asset_balance(asset):
    try:
        balance = client.get_asset_balance(asset=asset)
        return float(balance['free'])  # 'free' is the amount available for trading
    except Exception as e:
        logger.error(f"Error fetching balance for {asset}: {e}")
        return 0

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
        symbol_info = client.get_symbol_info(symbol)
        formatted_quantity = format_quantity(quantity, symbol_info)  # Define formatted_quantity here

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

# Data Processing Functions

# Prepares the data for model training and prediction.
def prepare_data(df, target_col='target', window=5, features_to_use=None):
    try:
        if 'target' not in df.columns:
            df['target'] = (df['close'].shift(-window) > df['close']).astype(int)

        if features_to_use is None:
            features_to_use = ['SMA', 'RSI', 'MACD', 'upper_band', 'EMA', 'ATR', 'slowk', 'CCI', 'SAR', 'OBV', 'Open', 'High', 'Low', 'Close', 'volume']

        assert all(feature in df.columns for feature in features_to_use), "Some features are missing in the DataFrame"

        labels = df[target_col]

        if class_imbalance_detected(labels):
            df, labels = apply_smote(df, features_to_use, labels)
            df = add_noise(df)
            df = apply_elastic_transformations(df)

        return df[features_to_use], labels
    except Exception as e:
        logger.error(f"Error in prepare_data: {e}")

def class_imbalance_detected(labels, threshold=0.3):
    class_counts = labels.value_counts(normalize=True)
    return any(class_counts < threshold)

def apply_smote(df, features, labels):
    smote = SMOTE()
    resampled_features, resampled_labels = smote.fit_resample(df[features], labels)
    resampled_df = pd.DataFrame(resampled_features, columns=features)
    resampled_df['target'] = resampled_labels
    return resampled_df, resampled_labels

# Adds random noise to the dataset to increase robustness.
def add_noise(df, noise_level=0.01):
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    noise = np.random.normal(0, noise_level, df[numerical_cols].shape)
    df[numerical_cols] += noise
    return df

def apply_elastic_transformations(df, scale_factor=1.05):
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] *= scale_factor
    return df

def oversample_minority_class(df, labels):
    oversampler = SMOTE(sampling_strategy='minority')
    features, labels = oversampler.fit_resample(df[['Open', 'High', 'Low', 'Close', 'Volume']], labels)
    df = pd.DataFrame(features, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    df['target'] = labels
    return df

def undersample_majority_class(df, labels):
    undersampler = RandomUnderSampler(sampling_strategy='majority')
    features, labels = undersampler.fit_resample(df[['Open', 'High', 'Low', 'Close', 'Volume']], labels)
    df = pd.DataFrame(features, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    df['target'] = labels
    return df

# Indicator and Strategy Functions
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

def calculate_signals(df, logistic_regression_model, scaler, features_to_use):
    
    # Calculate various signals for strategy selection.
    last_data = df[-1:].drop(columns=['target'])
    last_data_scaled = scaler.transform(last_data[features_to_use])

    logistic_regression_signal = logistic_regression_model.predict(last_data_scaled)[0]
    lorentzian_signal = lorentzian_classification(df)
    momentum_signal = squeezed_momentum_indicator(df)

    return logistic_regression_signal, lorentzian_signal, momentum_signal

def default_strategy_decision(logistic_regression_signal, lorentzian_signal, momentum_signal):
    logistic_regression_threshold = 0.5  # Adjust as needed
    lorentzian_threshold = 0.5  # Adjust as needed
    momentum_threshold = 0.5  # Adjust as needed

    combined_signal = (logistic_regression_signal > logistic_regression_threshold) + \
                    (lorentzian_signal > lorentzian_threshold) + \
                    (momentum_signal > momentum_threshold)

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
        action = "No Action"

    return chosen_strategy, action

def choose_strategy(df, logistic_regression_model, scaler, symbol_to_trade, features_to_use, market_data):
    logistic_regression_signal, lorentzian_signal, momentum_signal = calculate_signals(df, logistic_regression_model, scaler, features_to_use)

    if len(df) <= 1:
        return "Insufficient data for strategy selection", "Hold"

    # Fetch additional data
    coinmarketcap_data = get_coinmarketcap_data(coinmarketcap_api_key, symbol_to_trade)
    if coinmarketcap_data:
        market_cap = coinmarketcap_data['quote']['USD']['market_cap']
        volume_24h = coinmarketcap_data['quote']['USD']['volume_24h']
        percent_change_24h = coinmarketcap_data['quote']['USD']['percent_change_24h']
        print(f"Market Cap: {market_cap}")
        print(f"24h Volume: {volume_24h}")
        print(f"24h Change: {percent_change_24h}%")
        
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

def evaluate_and_choose_strategy(df, logistic_regression_model, scaler, symbol_to_trade, features_to_use):
    logistic_regression_signal, lorentzian_signal, momentum_signal = calculate_signals(df, logistic_regression_model, scaler, features_to_use)

    logistic_regression_threshold = 0.5  # Adjust as needed
    lorentzian_threshold = 0.5  # Adjust as needed
    momentum_threshold = 0.5  # Adjust as needed

    # Combine signals based on thresholds
    combined_signal = (logistic_regression_signal > logistic_regression_threshold) + \
                    (lorentzian_signal > lorentzian_threshold) + \
                    (momentum_signal > momentum_threshold)

    # Determine strategy and corresponding action (Buy or Sell)
    if combined_signal >= 2:
        chosen_strategy = "Combined Strategy"
        action = "Buy"
    elif logistic_regression_signal > logistic_regression_threshold:
        chosen_strategy = "Logistic Regression"
        action = "Buy"
    elif lorentzian_signal > lorentzian_threshold:
        chosen_strategy = "Lorentzian Classification"
        action = "Buy"
    elif momentum_signal > momentum_threshold:
        chosen_strategy = "Squeezed Momentum Indicator"
        action = "Buy"
    elif logistic_regression_signal < logistic_regression_threshold and lorentzian_signal < lorentzian_threshold and momentum_signal < momentum_threshold:
        chosen_strategy = "Bearish Market Indicators"
        action = "Sell"
    else:
        chosen_strategy = "No Clear Signal"
        action = "Hold"

    return chosen_strategy, action

def analyze_market_trend(df, short_window=12, long_window=26):
    if df['short_ema'].iloc[-1] > df['long_ema'].iloc[-1] and df['close'].iloc[-1] > df['short_ema'].iloc[-1]:
        return 'upward'
    elif df['short_ema'].iloc[-1] < df['long_ema'].iloc[-1] and df['close'].iloc[-1] < df['short_ema'].iloc[-1]:
        return 'downward'
    else:
        return 'sideways'

def calculate_market_volatility(df, window=14):
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=window).std()
    current_volatility = df['volatility'].iloc[-1]
    return current_volatility

# Model Training and Prediction Functions
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)
    return model

def make_trading_decision(df, rf_model, scaler, trained_features, window_size):
    # Ensure the features and window size are defined
    if trained_features is None or window_size is None:
        raise ValueError("Trained features and window size must be provided")

    # Prepare data
    features, labels = prepare_data(df, window=window_size, features_to_use=trained_features)
    scaler = StandardScaler().fit(features)

    # Get the latest data for prediction
    current_data = df[-1:][trained_features]
    current_data_scaled = scaler.transform(current_data)

    # Predict the strategy index using the random forest model
    predicted_strategy_index = rf_model.predict(current_data_scaled)[0]

    # Define the names of the strategies corresponding to the indices
    strategy_names = ["Logistic Regression", "Lorentzian Classification", "Squeezed Momentum Indicator"]

    # Get the name of the chosen strategy based on the predicted index
    chosen_strategy = strategy_names[predicted_strategy_index]

    # Get the current price from the DataFrame
    current_price = df['close'].iloc[-1]

    # Make a trading decision based on the predicted strategy index and current price
    if predicted_strategy_index > current_price * 1.01:  # Example threshold for a buy decision
        return "Buy", chosen_strategy
    elif predicted_strategy_index < current_price * 0.99:  # Example threshold for a sell decision
        return "Sell", chosen_strategy
    else:
        return "Hold", chosen_strategy

def predict_and_trade(model, X):
    prediction = model.predict(X)
    return prediction

# Technical Indicators for Random Forest Regressor
def add_technical_indicators(df):
    # Simple Moving Average
    df['SMA'] = talib.SMA(df['close'], timeperiod=20)
    # Relative Strength Index
    df['RSI'] = talib.RSI(df['close'])
    # Moving Average Convergence Divergence
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'])
    # Bollinger Bands
    df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(df['close'], timeperiod=20)
    # Exponential Moving Average
    df['EMA'] = talib.EMA(df['close'], timeperiod=20)
    # Average True Range
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    # Stochastic Oscillator
    df['slowk'], df['slowd'] = talib.STOCH(df['high'], df['low'], df['close'])
    # Commodity Channel Index
    df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
    # Parabolic SAR
    df['SAR'] = talib.SAR(df['high'], df['low'])
    # On-Balance Volume
    df['OBV'] = talib.OBV(df['close'], df['volume'])
    return df

# Define evaluation functions
def is_lr_model_reliable(lr_signal):
    # Implement your logic here
    return True  # Placeholder

def is_lorentzian_signal_strong(lorentzian_signal):
    # Implement your logic here
    return True  # Placeholder

def is_squeezed_momentum_signal_strong(squeezed_momentum_signal):
    # Implement your logic here
    return True  # Placeholder

def main():
    print("Trading Bot v.1 TETRAHEDRON ALPHA")
    symbol_to_trade, interval, window_size, investment_amount = get_user_input()
    quote_asset = get_quote_asset(symbol_to_trade)

    # Fetching and preparing initial data
    klines = get_symbol_klines(symbol_to_trade, interval, 100)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.set_index('timestamp', inplace=True)

    # Adding technical indicators
    df_with_indicators = add_technical_indicators(df)

    executed_order = False  # Track whether a buy/sell order has been executed
    chosen_strategy = "No Strategy Selected"  # Initialize chosen_strategy

    try:
        rf_model = joblib.load('random_forest_model.pkl')
    except Exception as e:
        logger.error(f"Error loading Random Forest model: {e}")
        rf_model = None

    try:
        logistic_regression_model = joblib.load('logistic_regression_model.pkl')
    except Exception as e:
        logger.error(f"Error loading Logistic Regression model: {e}")
        logistic_regression_model = None
    
    while not executed_order:
        try:
            # Fetch new market data and prepare features
            klines = get_symbol_klines(symbol_to_trade, interval, 100)
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df.set_index('timestamp', inplace=True)
            df_with_indicators = add_technical_indicators(df)

            # Prepare data for model
            features, _ = prepare_data(df_with_indicators, window=window_size)

            # Initialize and fit the scaler with the features
            scaler = StandardScaler().fit(features)

            if rf_model:
                # Get the latest data for prediction
                current_data = features.iloc[-1:]
                current_data_scaled = scaler.transform(current_data)

                # Make trading decision
                action, chosen_strategy = make_trading_decision(df_with_indicators, rf_model, scaler, features.columns.tolist(), window_size)
                print(f"Chosen Strategy: {chosen_strategy}, Action: {action}")

                # Analysis Results
                if logistic_regression_model:
                    try:
                        current_data_scaled = scaler.transform(features.iloc[-1:])
                        lr_signal = logistic_regression_model.predict(current_data_scaled)[0]
                        lorentzian_signal = lorentzian_classification(df_with_indicators)
                        squeezed_momentum_signal = squeezed_momentum_indicator(df_with_indicators)

                        # Evaluate and Choose Strategy
                        if is_lr_model_reliable(lr_signal):
                            chosen_strategy = "Logistic Regression"
                        elif is_lorentzian_signal_strong(lorentzian_signal):
                            chosen_strategy = "Lorentzian Classification"
                        elif is_squeezed_momentum_signal_strong(squeezed_momentum_signal):
                            chosen_strategy = "Squeezed Momentum Indicator"
                        else:
                            chosen_strategy = "Default Strategy"

                        print("\nAnalysis Results:")
                        print(f"Chosen Strategy: {chosen_strategy}")
                        print(f"Logistic Regression Signal: {lr_signal}")
                        print(f"Lorentzian Signal: {lorentzian_signal}")
                        print(f"Squeezed Momentum Signal: {squeezed_momentum_signal}")
                        logger.info(f"Chosen Strategy: {chosen_strategy}")
                    except Exception as e:
                        logger.error(f"Error in analysis results: {e}")
                
                # Execute order based on the decision
                if action in ["Buy", "Sell"]:
                    quantity = calculate_order_quantity(investment_amount, df_with_indicators['close'].iloc[-1], quote_asset)
                    order = place_order(symbol_to_trade, Client.SIDE_BUY if action == "Buy" else Client.SIDE_SELL, quantity)
                    if order:
                        executed_order = True
                        print(f"Order executed: {order}")
                elif action == "No Action":
                    print("No trading action taken. Holding and gathering data.")
            else:
                # Fallback strategy decision
                chosen_strategy, action = default_strategy_decision()
                print(f"Chosen Strategy: {chosen_strategy}, Action: {action}")

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            time.sleep(60)  # Sleep before retrying

        # Additional analysis and logging
        # ...

    # Additional code for the main function
    # ...

if __name__ == "__main__":
    main()
