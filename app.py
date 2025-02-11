import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    X = df.iloc[:, 3:13]
    y = df.iloc[:, -1]
    X = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def train_model(X_train, y_train):
    model = Sequential([
        Dense(11, activation='relu'),
        Dense(7, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)
    return model

def main():
    st.title("Customer Churn Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        X, y, scaler = preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34)
        model = train_model(X_train, y_train)
        st.write("Model Training Completed.")
        
        st.sidebar.header("User Input Features")
        input_data = []
        for col in df.columns[3:13]:
            input_data.append(st.sidebar.number_input(col, value=float(df[col].mean())))
        
        if st.sidebar.button("Predict"):
            input_scaled = scaler.transform([input_data])
            prediction = model.predict(input_scaled)
            churn = "Yes" if prediction >= 0.5 else "No"
            st.write(f"Prediction: Customer Churn? {churn}")
