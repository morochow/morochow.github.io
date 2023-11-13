import pandas as pd
import joblib
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Function to read and preprocess the data
def read_and_preprocess(file_path):
    df = pd.read_csv(file_path, sep=';')
    df['timeOpen'] = pd.to_datetime(df['timeOpen'])
    df.set_index('timeOpen', inplace=True)
    return df

# Feature Engineering
def add_technical_indicators(df):
    df['SMA'] = talib.SMA(df['close'], timeperiod=20)
    df['RSI'] = talib.RSI(df['close'])
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'])
    df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(df['close'], timeperiod=20)
    df['EMA'] = talib.EMA(df['close'], timeperiod=20)
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['slowk'], df['slowd'] = talib.STOCH(df['high'], df['low'], df['close'])
    df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
    df['SAR'] = talib.SAR(df['high'], df['low'])
    df['OBV'] = talib.OBV(df['close'], df['volume'])
    return df

# Load and preprocess data
file_path = 'Bitcoin_6_24_2010-8_23_2010_historical_data_coinmarketcap.csv'
df = read_and_preprocess(file_path)
df = add_technical_indicators(df)
df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
df.dropna(inplace=True)

# Select only numeric columns for features
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
X = df[numeric_cols]
if 'Target' in X.columns:
    X = X.drop('Target', axis=1)
y = df['Target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
for i in tqdm(range(100), desc="Training Progress"):
    time.sleep(0.1)  # Simulate a part of the training process
    if i == 0:
        model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Cross-Validation
scores = cross_val_score(model, X, y, cv=5)
print("Cross-validated scores:", scores)
print("Average score:", scores.mean())

# Save the model
joblib.dump(model, 'forest_model1.pkl')