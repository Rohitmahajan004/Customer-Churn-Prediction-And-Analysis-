"""
Streamlit Churn Prediction App

Place this file (streamlit_churn_app.py) in your project folder along with:
- best_churn_model.pkl  (trained model saved from the notebook)
- (optional) ecommerce_churn.csv  (transactional CSV) or an aggregated customer-level CSV

How it works:
- Upload either a transactional CSV (with columns including 'Customer ID' and 'Purchase Date')
  or a pre-aggregated customer-features CSV with the exact features list used in the notebook.
- The app will automatically detect transactional data and aggregate it to customer-level features.
- You can preview data, run bulk predictions, or use the manual input form to test single customers.

Requirements (also include these in your requirements.txt for deployment):
streamlit
pandas
numpy
scikit-learn
joblib
xgboost  # optional, only if model uses xgboost
matplotlib

Run locally:
> pip install -r requirements.txt
> streamlit run streamlit_churn_app.py

"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title='E-commerce Churn Predictor', layout='wide')

FEATURES = [
    'total_spent','avg_order_value','max_order_value','num_orders','orders_per_month',
    'avg_items_per_order','return_rate','num_categories','days_since_last_purchase','age',
    'gender','payment_method'
]

@st.cache_data
def load_model(path='models/best_churn_model.pkl'):
    try:
        m = joblib.load(path)
        return m
    except Exception as e:
        return None

@st.cache_data
def aggregate_transactions(df):
    # Ensure proper types
    df.columns = [c.strip() for c in df.columns]
    if 'Purchase Date' in df.columns:
        df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], errors='coerce')
    # Numeric coercion
    for col in ['Product Price','Quantity','Total Purchase Amount','Customer Age','Returns','Churn']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    agg_funcs = {
        'Total Purchase Amount': ['sum','mean','max'],
        'Purchase Date': ['min','max','count'],
        'Quantity': 'sum',
        'Returns': 'sum'
    }
    cust = df.groupby('Customer ID').agg(agg_funcs)
    cust.columns = ['_'.join(col).strip() for col in cust.columns.values]
    cust = cust.rename(columns={
        'Total Purchase Amount_sum': 'total_spent',
        'Total Purchase Amount_mean': 'avg_order_value',
        'Total Purchase Amount_max': 'max_order_value',
        'Purchase Date_min': 'first_purchase_date',
        'Purchase Date_max': 'last_purchase_date',
        'Purchase Date_count': 'num_orders',
        'Quantity_sum': 'total_quantity',
        'Returns_sum': 'total_returns'
    })
    dataset_max_date = df['Purchase Date'].max()
    cust['days_since_last_purchase'] = (dataset_max_date - pd.to_datetime(cust['last_purchase_date'])).dt.days
    cust['return_rate'] = cust['total_returns'] / cust['num_orders']
    category_counts = df.groupby('Customer ID')['Product Category'].nunique().rename('num_categories')
    cust = cust.merge(category_counts, left_index=True, right_index=True, how='left')
    payment_mode = df.groupby('Customer ID')['Payment Method'].agg(lambda x: x.mode().iloc[0] if len(x.mode())>0 else np.nan).rename('payment_method')
    cust = cust.merge(payment_mode, left_index=True, right_index=True, how='left')
    age = df.groupby('Customer ID')['Customer Age'].agg(lambda x: x.iloc[0]).rename('age')
    gender = df.groupby('Customer ID')['Gender'].agg(lambda x: x.mode().iloc[0] if len(x.mode())>0 else np.nan).rename('gender')
    name = df.groupby('Customer ID')['Customer Name'].agg(lambda x: x.iloc[0]).rename('customer_name')
    cust = cust.merge(age, left_index=True, right_index=True, how='left')
    cust = cust.merge(gender, left_index=True, right_index=True, how='left')
    cust = cust.merge(name, left_index=True, right_index=True, how='left')
    # churn label if present
    if 'Churn' in df.columns:
        churn_label = df.groupby('Customer ID')['Churn'].max().rename('churn')
        cust = cust.merge(churn_label, left_index=True, right_index=True, how='left')
    cust['orders_per_month'] = cust['num_orders'] / ((pd.to_datetime(cust['last_purchase_date']) - pd.to_datetime(cust['first_purchase_date'])).dt.days/30.0 + 0.001)
    cust['avg_items_per_order'] = cust['total_quantity'] / cust['num_orders']
    cust.replace([np.inf, -np.inf], np.nan, inplace=True)
    cust['return_rate'] = cust['return_rate'].fillna(0)
    return cust.reset_index()

@st.cache_data
def prepare_features(df):
    # Keep only features needed for model
    X = df.copy()
    missing_cols = [c for c in FEATURES if c not in X.columns]
    if missing_cols:
        # fill missing numeric with zeros and categorical with 'Unknown'
        for c in missing_cols:
            if c in ['gender','payment_method']:
                X[c] = 'Unknown'
            else:
                X[c] = 0
    return X[FEATURES]

st.title('🛒 E-commerce Customer Churn Predictor')
st.markdown('Upload transactional CSV or pre-aggregated customer features CSV to run predictions.')

# Sidebar: model load and sample data
st.sidebar.header('Model & Data')
model_path = st.sidebar.text_input('Model file path', value='best_churn_model.pkl')
model = load_model(model_path)
if model is None:
    st.sidebar.error('Model not found at the path specified. Place best_churn_model.pkl in app folder or change path.')
else:
    st.sidebar.success('Model loaded')

use_sample = st.sidebar.checkbox('Use sample input (two examples)', value=False)

uploaded_file = st.file_uploader('Upload CSV (transactional or customer-level)', type=['csv'])

# Main layout: two columns
col1, col2 = st.columns([2,1])

with col1:
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error('Error reading CSV: '+str(e))
            st.stop()
        st.subheader('Raw data preview')
        st.dataframe(df_raw.head())

        # Detect transactional vs aggregated
        if 'Purchase Date' in df_raw.columns and 'Customer ID' in df_raw.columns:
            st.info('Detected transactional data — aggregating to customer level...')
            cust_df = aggregate_transactions(df_raw)
            st.subheader('Aggregated customer-level preview')
            st.dataframe(cust_df.head())
        else:
            st.info('Assuming uploaded CSV is already customer-level features')
            cust_df = df_raw.copy()

        # Prepare features
        X = prepare_features(cust_df)
        st.subheader('Features used for prediction')
        st.write(X.head())

        if model is not None:
            if st.button('Run bulk predictions'):
                try:
                    preds = model.predict(X)
                    probs = model.predict_proba(X)[:,1] if hasattr(model, 'predict_proba') else None
                    cust_df['predicted_churn'] = preds
                    if probs is not None:
                        cust_df['churn_probability'] = probs
                    st.success('Predictions complete')
                    st.dataframe(cust_df.head(50))
                    csv = cust_df.to_csv(index=False)
                    st.download_button('Download predictions CSV', data=csv, file_name='predictions.csv')
                except Exception as e:
                    st.error('Error during prediction: '+str(e))
    else:
        st.info('No CSV uploaded — you can test the model using the manual input form on the right.')

    # Quick EDA if model present and sample data
    if model is not None and uploaded_file is not None:
        if 'predicted_churn' in locals() or 'predicted_churn' in globals():
            try:
                fig, ax = plt.subplots(figsize=(6,3))
                cust_df['predicted_churn'].value_counts().sort_index().plot(kind='bar', ax=ax)
                ax.set_xticklabels(['Retained(0)','Churned(1)'])
                ax.set_title('Predicted churn distribution')
                st.pyplot(fig)
            except Exception:
                pass

with col2:
    st.subheader('Manual customer test')
    if use_sample:
        # Sample pick
        sample_choice = st.selectbox('Choose sample', ['Active customer (retained likely)','Inactive customer (churn likely)'])
        if sample_choice.startswith('Active'):
            example = {
                'total_spent': 2500,
                'avg_order_value': 250,
                'max_order_value': 500,
                'num_orders': 10,
                'orders_per_month': 2,
                'avg_items_per_order': 3,
                'return_rate': 0.1,
                'num_categories': 5,
                'days_since_last_purchase': 5,
                'age': 32,
                'gender': 'Female',
                'payment_method': 'Credit Card'
            }
        else:
            example = {
                'total_spent': 200,
                'avg_order_value': 50,
                'max_order_value': 60,
                'num_orders': 4,
                'orders_per_month': 0.2,
                'avg_items_per_order': 1,
                'return_rate': 0.5,
                'num_categories': 1,
                'days_since_last_purchase': 300,
                'age': 45,
                'gender': 'Male',
                'payment_method': 'PayPal'
            }
        for k,v in example.items():
            if isinstance(v, (int,float)):
                example[k] = st.number_input(k, value=float(v))
            else:
                example[k] = st.text_input(k, value=str(v))
    else:
        # manual inputs
        example = {}
        st.write('Fill values to test a single customer')
        example['total_spent'] = st.number_input('total_spent', value=500.0)
        example['avg_order_value'] = st.number_input('avg_order_value', value=100.0)
        example['max_order_value'] = st.number_input('max_order_value', value=150.0)
        example['num_orders'] = st.number_input('num_orders', value=5)
        example['orders_per_month'] = st.number_input('orders_per_month', value=0.5)
        example['avg_items_per_order'] = st.number_input('avg_items_per_order', value=2.0)
        example['return_rate'] = st.number_input('return_rate', min_value=0.0, max_value=1.0, value=0.1)
        example['num_categories'] = st.number_input('num_categories', value=2)
        example['days_since_last_purchase'] = st.number_input('days_since_last_purchase', value=30)
        example['age'] = st.number_input('age', value=30)
        example['gender'] = st.selectbox('gender', ['Male','Female','Unknown'])
        example['payment_method'] = st.selectbox('payment_method', ['Credit Card','PayPal','Debit Card','Unknown'])

    if model is None:
        st.warning('No model loaded — upload model file to the app folder or set correct model path in sidebar')
    else:
        if st.button('Predict single customer'):
            row = pd.DataFrame([example])
            Xrow = prepare_features(row)
            try:
                pred = model.predict(Xrow)[0]
                prob = model.predict_proba(Xrow)[:,1][0] if hasattr(model, 'predict_proba') else None
                st.write('**Prediction:**', 'Churn' if int(pred)==1 else 'Retained')
                if prob is not None:
                    st.write('**Churn probability:**', round(float(prob),3))
            except Exception as e:
                st.error('Prediction error: '+str(e))

st.markdown('---')
st.caption('App created by ChatGPT — drop me a message if you want extra features like SHAP explanations, dashboards, or Streamlit Cloud deployment files.')
