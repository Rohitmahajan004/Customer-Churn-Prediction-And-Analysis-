from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, jsonify


app = Flask(__name__)

# Load Model and Scaler
with open('churn_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load the column names used during training
with open('training_columns.pkl', 'rb') as file:
    training_columns = pickle.load(file)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/analysis')
def analysis_page():
    return render_template('analysis.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    data = request.form
    features = {
        'Purchase_Frequency': float(data['Purchase_Frequency']),
        'Average_Order_Value': float(data['Average_Order_Value']),
        'Time_Between_Purchases': float(data['Time_Between_Purchases']),
        'Lifetime_Value': float(data['Lifetime_Value']),
        'Preferred_Purchase_Times': float(data['Preferred_Purchase_Times']),
        # Add any other form data here
    }

    # Convert to DataFrame for easier manipulation (like encoding)
    input_df = pd.DataFrame([features])

    # Apply the same transformations to input data as was done during training
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Ensure the columns of the input match the model's training columns
    input_df = input_df.reindex(columns=training_columns, fill_value=0)

    # Scale the features using the same scaler as in training
    scaled_data = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0, 1]

    result = "Churn" if prediction[0] == 1 else "Not Churn"
    return jsonify({'result': result, 'probability': f"{probability * 100:.2f}%"})

if __name__ == '__main__':
    app.run(debug=True)
