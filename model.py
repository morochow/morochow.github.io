import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import talib

# Function to fetch data from CoinMarketCap
def fetch_data(symbol, start_date, end_date, coinmarketcap_api_key):
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
    parameters = {
        'symbol': symbol,
        'time_start': start_date,
        'time_end': end_date,
        'convert': 'USD'
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': coinmarketcap_api_key,
    }

    response = requests.get(url, headers=headers, params=parameters)
    data = response.json()

    # Parse the data into a DataFrame
    df = pd.DataFrame([{
        'Open': day['quote']['USD']['open'],
        'High': day['quote']['USD']['high'],
        'Low': day['quote']['USD']['low'],
        'Close': day['quote']['USD']['close'],
        'Volume': day['quote']['USD']['volume']
    } for day in data['data']['quotes']])

    df.index = pd.to_datetime([day['time_open'] for day in data['data']['quotes']])
    return df

# Feature Engineering
def add_technical_indicators(df):
    # Simple Moving Average
    df['SMA'] = talib.SMA(df['Close'], timeperiod=20)
    # Relative Strength Index
    df['RSI'] = talib.RSI(df['Close'])
    # Moving Average Convergence Divergence
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['Close'])
    # Bollinger Bands
    df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(df['Close'], timeperiod=20)
    # Exponential Moving Average
    df['EMA'] = talib.EMA(df['Close'], timeperiod=20)
    # Average True Range
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    # Stochastic Oscillator
    df['slowk'], df['slowd'] = talib.STOCH(df['High'], df['Low'], df['Close'])
    # Commodity Channel Index
    df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=20)
    # Parabolic SAR
    df['SAR'] = talib.SAR(df['High'], df['Low'])
    # On-Balance Volume
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])
    return df

# Load Data
coinmarketcap_api_key = "e75fb4c4-0e8c-49e5-b18c-941df88edc0b"
symbol = "BTCUSDT"  # Example symbol
start_date = "2022-01-01"
end_date = "2023-11-12"
df = fetch_data(symbol, start_date, end_date, coinmarketcap_api_key)

# Add technical indicators
df = add_technical_indicators(df)

# Define target variable
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Drop rows with NaN values
df.dropna(inplace=True)

# Split data into features and target
X = df.drop('Target', axis=1)
y = df['Target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Save the model
joblib.dump(model, 'random_forest_model.pkl')
