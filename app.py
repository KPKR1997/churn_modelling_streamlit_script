import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

MODEL_PATH = "models/churn_model.h5"
SCALER_PATH = "models/scaler.pkl"

def load_model_and_scaler():
    model = load_model(MODEL_PATH, compile=False)  # Load model once
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def preprocess_input(input_dict, scaler):
    # Convert categorical text inputs to numerical values
    geography_map = {"France": [1, 0], "Germany": [0, 1], "Spain": [0, 0]}
    gender_map = {"Male": 1, "Female": 0}
    
    processed_input = [
        input_dict["CreditScore"],
        input_dict["Age"],
        input_dict["Tenure"],
        input_dict["Balance"],
        input_dict["NumOfProducts"],
        input_dict["HasCrCard"],
        input_dict["IsActiveMember"],
        input_dict["EstimatedSalary"]
    ] + geography_map[input_dict["Geography"]] + [gender_map[input_dict["Gender"]]]
    
    # Normalize input data
    processed_input = scaler.transform([processed_input])
    return processed_input

def main():
    st.title("Customer Churn Prediction")
    model, scaler = load_model_and_scaler()

    st.subheader("ANN model using two hidden layers with ReLU activation and output layer with sigmoid function. Epochs : 50")
    
    st.sidebar.header("User Input Features")
    input_dict = {}
    
    input_dict["CreditScore"] = st.sidebar.number_input("Credit Score", min_value=300, max_value=900, value=650)
    input_dict["Age"] = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35)
    input_dict["Tenure"] = st.sidebar.number_input("Tenure", min_value=0, max_value=10, value=5)
    input_dict["Balance"] = st.sidebar.number_input("Balance", min_value=0.0, max_value=300000.0, value=50000.0)
    input_dict["NumOfProducts"] = st.sidebar.number_input("Number of Products", min_value=1, max_value=4, value=1)
    
    # Use radio buttons for Yes/No input and convert to 1 (Yes) or 0 (No)
    input_dict["HasCrCard"] = 1 if st.sidebar.radio("Has Credit Card?", ['No', 'Yes']) == 'Yes' else 0
    input_dict["IsActiveMember"] = 1 if st.sidebar.radio("Is Active Member?", ['No', 'Yes']) == 'Yes' else 0
    
    input_dict["EstimatedSalary"] = st.sidebar.number_input("Estimated Salary", min_value=0.0, max_value=200000.0, value=50000.0)
    input_dict["Geography"] = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
    input_dict["Gender"] = st.sidebar.selectbox("Gender", ["Male", "Female"])
    
    if st.sidebar.button("Predict"):
        input_processed = preprocess_input(input_dict, scaler)
        prediction = model.predict(input_processed)
        churn = "Positive" if prediction >= 0.5 else "Negative"
        st.write(f"Prediction: Customer Churn: {churn}")

if __name__ == "__main__":
    main()
