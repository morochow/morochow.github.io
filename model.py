import pandas as pd
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from tqdm import tqdm
import time
import numpy as np

# Function to load data from a CSV file
def load_data(file_path):
    df = pd.read_csv(file_path, sep=';', parse_dates=['timeOpen', 'timeClose', 'timeHigh', 'timeLow', 'timestamp'], dayfirst=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    df.set_index('timeOpen', inplace=True)
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
file_path = "Bitcoin_6_24_2010-8_23_2010_historical_data_coinmarketcap.csv"
df = load_data(file_path)

# Add technical indicators
df = add_technical_indicators(df)

# Define target variable
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Drop rows with NaN values
df.dropna(inplace=True)

# Split data into features and target
X = df.drop('Target', axis=1).select_dtypes(include=[np.number])
y = df['Target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
for i in tqdm(range(100), desc="Training Progress"):
    time.sleep(0.1)  # Simulate a part of the training process
    if i == 0:
        model.fit(X_train, y_train)  # Fit the model only once

# Evaluate the model
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Save the model
joblib.dump(model, 'random_forest_model2.pkl')
