﻿# Customer-Churn-Prediction-And-Analysis-
Steps to Create and Use the Project:
1. Prepare the Dataset
Collect and preprocess customer churn data.
Handle missing values, encode categorical variables, and scale numerical features.
Split the dataset into training and testing sets for model training and evaluation.
2. Train the Random Forest Model
Use the Random Forest algorithm to create a predictive model.
Evaluate the model using metrics like accuracy, precision, recall, and F1-score.
Save the trained model as a .pkl file for reuse in the Flask application.
3. Set Up the Flask Application
Create a Flask app with two pages:
Prediction Page: Allows users to input data and predict whether a customer will churn.
Analysis Page: Displays Tableau visualizations for customer churn analysis.
Organize the project folder with templates for HTML pages and static assets for CSS/JavaScript.
4. Create HTML Templates
Dashboard Page:
Embed Tableau visualizations using an iframe.
Prediction Page:
Add input fields for user data and display predictions dynamically.
5. Integrate Tableau
Create and publish a dashboard in Tableau (e.g., Tableau Public or Tableau Server).
Embed the Tableau visualization into the dashboard page of your Flask app using an iframe link.
6. Test the Application
Launch the Flask server and verify:
Predictions are accurate and user-friendly.
Tableau dashboard is embedded and interactive.
7. Use the Project
Provide the project folder, including the .pkl model file, Flask app, and necessary instructions to run the application.
Share a requirements.txt file for installing dependencies.


Install Python
Ensure Python is installed on your system (preferably version 3.7 or above).
Verify by running:
bash
Copy code
python --version
2. Install Flask
Install Flask using pip:
bash
Copy code
pip install flask
3. Organize Your Project
Create a project folder with the following structure:
scss
Copy code
project/
├── templates/  (contains HTML files)
│   ├── dashboard.html
│   ├── prediction.html
├── static/  (optional, for CSS/JS files)
├── churn_model.pkl  (your trained Random Forest model file)
├── app.py  (your main Flask application script)
├── requirements.txt  (list of dependencies)
4. Write requirements.txt
Create a file named requirements.txt in your project directory and include all dependencies:
Copy code
flask
numpy
scikit-learn
5. Activate Virtual Environment (Optional but Recommended)
Create a virtual environment to avoid conflicts:
bash
Copy code
python -m venv venv
Activate the virtual environment:
Windows:
bash
Copy code
venv\Scripts\activate
Mac/Linux:
bash
Copy code
source venv/bin/activate
Install dependencies in the virtual environment:
bash
Copy code
pip install -r requirements.txt
6. Run the Flask App
Navigate to the project directory:
bash
Copy code
cd project
Start the Flask server:
bash
Copy code
python app.py
If the app is configured correctly, you'll see output similar to:
csharp
Copy code
* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
7. Access the Web Application
Open a web browser and visit:
arduino
Copy code
http://127.0.0.1:5000/
8. Test the Functionality
On the Prediction Page, input data and check predictions.
On the Analysis Page, view Tableau visualizations.
9. Troubleshooting
If the app doesn't run:
Check for errors in app.py or missing dependencies.
Ensure the model file (churn_model.pkl) exists in the project directory.
10. Stop the App
Press CTRL+C in the terminal to stop the Flask server.
