📊 Customer Churn Prediction & Analysis
This project predicts customer churn for an e-commerce platform and provides analysis of customer behavior. It uses machine learning to classify customers as "likely to churn" or "likely to stay," based on features such as last purchase date, age, gender, and payment method.

The project also includes an interactive Streamlit web application to test predictions using either manual inputs or uploaded customer data.

🚀 Features
Data preprocessing for handling categorical and numerical variables

Model training with Random Forest Classifier

Model saving & loading with joblib

Interactive Streamlit UI for predictions

Manual input testing for single customers

CSV upload for bulk predictions

Visualization & EDA capabilities (optional extensions)

📂 Project Structure
📁 customer-churn-prediction
│
├── app.py                   # Streamlit web application
├── train_and_save_model.py  # Script to train and save ML model
├── best_churn_model.pkl     # Saved trained model
├── ecommerce_customer_churn.csv  # Dataset
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation



Installation
Clone the repository
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction


Install dependencies

pip install -r requirements.txt

streamlit run app.py

📊 Dataset
The dataset is a synthetic e-commerce customer dataset generated using the Faker library.

Columns:
Customer ID: Unique identifier

Customer Age: Age of the customer

Gender: Male/Female

Purchase Date: Date of purchase

Product Category: Category of purchased product

Product Price: Price of product

Quantity: Quantity purchased

Total Purchase Amount: Transaction total

Payment Method: Credit Card, PayPal, etc.

Returns: 0 = No return, 1 = Returned

Churn: Target variable (0 = Retained, 1 = Churned)


odel Training
The Random Forest Classifier is used for prediction.
You can retrain the model by running:

python train_and_save_model.py

🌐 Web Application
The Streamlit app allows you to:

Manually input customer details and get a churn prediction

Upload a CSV file for bulk predictions

View sample predictions using predefined data


Requirements
Python 3.8+

pandas

scikit-learn

joblib

streamlit



