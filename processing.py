import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Load Dataset
df = pd.read_csv('data.csv')  # Replace with your dataset file name

# Ensure proper encoding of categorical columns
X = df.drop(['Churn_Probability', 'Customer_ID', 'Transaction_ID'], axis=1)
y = df['Churn_Probability']

# Categorize target if it is continuous
y = (y >= 0.5).astype(int)  # Binary classification (threshold: 0.5)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle Missing Values (if any)
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

# Encode Categorical Features
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Align Columns in train and test data
X_train, X_test = X_train.align(X_test, join='left', axis=1)

# Ensure that all missing values are filled after alignment
X_test.fillna(0, inplace=True)

# Scale Features
scaler = StandardScaler()

# Fit scaler on training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate Model
y_pred = model.predict(X_test_scaled)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save Model and Scaler
with open('churn_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

with open('training_columns.pkl', 'wb') as file:
    pickle.dump(X_train.columns, file)

