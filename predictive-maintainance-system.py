import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the CSV file
df = pd.read_csv('Datasheet.csv')

# Converting datetime columns to numeric values
df['Usage-Start_Date'] = pd.to_datetime(df['Usage-Start_Date']).apply(lambda x: x.timestamp())
df['Usage-End_Date'] = pd.to_datetime(df['Usage-End_Date']).apply(lambda x: x.timestamp())
df['Maintenance-Date'] = pd.to_datetime(df['Maintenance-Date']).apply(lambda x: x.timestamp())

# Defining the feature columns
feature_cols = ['Usage-Start_Date', 'Usage-End_Date', 'Usage-Hours', 'Maintenance-Date', 'Maintenance-Type', 'Maintenance-Hours', 'Sensor_Reading-Temperature', 'Sensor_Reading-Pressure', 'Sensor_Reading-Vibration']

# Defining the target column
target_col = 'Failure-Type'

# Encoding the categorical columns
le = LabelEncoder()
df['Maintenance-Type'] = le.fit_transform(df['Maintenance-Type'])
df['Failure-Type'] = le.fit_transform(df['Failure-Type'])

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], df[target_col], test_size=0.2, random_state=42)

# Training a Random Forest Classifier on the training data
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Predictions on the testing data
y_pred = rfc.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))