import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Load dataset
df = pd.read_csv("datasets/Churn_Modelling.csv")

# Data preprocessing
X = df.iloc[:, 3:13]
y = df.iloc[:, -1]

# Convert categorical features to numeric
X = pd.get_dummies(X, drop_first=True)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=34)

# Define ANN model
model = Sequential([
    Dense(11, activation='relu'),
    Dense(7, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)

# Save model and preprocessor
model.save("models/churn_model.h5")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model and Scaler Saved Successfully!")
