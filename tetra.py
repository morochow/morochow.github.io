import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import StratifiedKFold

cross_val = StratifiedKFold(n_splits=5)

# Load your data with semicolon delimiter
df = pd.read_csv('Bitcoin_6_24_2010-8_23_2010_historical_data_coinmarketcap.csv', delimiter=';')

# Convert the 'timestamp' column to datetime-like object
df['timestamp'] = pd.to_datetime(df['timestamp'])

df['action'] = 'hold'  # By default, set "hold" for all rows
df.loc[df['close'] > df['open'], 'action'] = 'buy'  # Buy when closing price is higher than opening price
df.loc[df['close'] < df['open'], 'action'] = 'sell'  # Sell when closing price is lower than opening price
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['hour'] = df['timestamp'].dt.hour

X = df.drop(['action', 'timestamp'], axis=1)
# Define your target variable and features
y = df['action']  # Use the 'action' column as the target variable
X = df.drop(['action', 'timestamp'], axis=1)  # Drop the 'action' and 'timestamp' columns to use the remaining columns as features

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define hyperparameters for tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],  # Updated 'max_features' parameter
    'bootstrap': [True, False]
}

# Initialize the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Select numeric columns from X_train
X_train_numeric = X_train.select_dtypes(include=['number'])

# Fit the grid search to the data
grid_search.fit(X_train_numeric, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:")
print(best_params)

# Train a RandomForestClassifier with the best hyperparameters
best_rf = RandomForestClassifier(**best_params, random_state=42)
best_rf.fit(X_train, y_train)

# Predict on the test set
predictions = best_rf.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, predictions))

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, best_rf.predict_proba(X_test)[:,1])
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

# Save the model
with open('rsaa.pkl', 'wb') as file:
    pickle.dump(best_rf, file)

print("Hyperparameter tuning, model training, and evaluation complete. Model saved as 'rsaa.pkl'")
