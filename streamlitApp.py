import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your trained model
model = joblib.load('xgboost_model.joblib')

def predict(input_data):
    """Function to predict using the ML model."""
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return prediction

def main():
    # Title of the application
    st.title('Polyfunctionality Prediction App')

    # Input features from the user
    feature1 = st.text_input('Enter Feature 1 (e.g., huADCP)')
    feature2 = st.text_input('Enter Feature 2 (e.g., mADCP)')
    feature3 = st.text_input('Enter Feature 3 (e.g., huADNP)')
    # Continue adding input features according to your model

    # Button to make prediction
    if st.button('Predict Polyfunctionality'):
        input_data = [feature1, feature2, feature3]  # Adjust according to your input features
        prediction = predict(input_data)
        st.success(f'The predicted polyfunctionality is {prediction[0]}')

if __name__ == '__main__':
    main()
