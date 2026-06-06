# Customer Churn Prediction Using Machine Learning
## Project Overview
This project predicts whether a telecom customer is likely to churn (leave the service) using Machine Learning techniques. The project includes data cleaning, exploratory data analysis (EDA), feature engineering, model training, evaluation, and deployment preparation.

## Objectives
* Analyze customer behavior and churn patterns.
* Identify important factors affecting customer churn.
* Build and compare Machine Learning models.
* Predict customer churn with high accuracy.
---
## Dataset
**Dataset:** Telco Customer Churn Dataset
The dataset contains customer demographic information, account details, services subscribed, billing information, and churn status.
### Key Features
* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure
* Phone Service
* Internet Service
* Contract Type
* Payment Method
* Monthly Charges
* Total Charges
* Churn (Target Variable)
---
## Exploratory Data Analysis (EDA)
Performed:
* Missing value analysis
* Churn distribution analysis
* Contract type analysis
* Tenure distribution analysis
* Monthly charges analysis
* Correlation analysis
* Feature importance visualization
### Key Findings
* Customers with month-to-month contracts are more likely to churn.
* Customers with shorter tenure show higher churn rates.
* Monthly charges significantly influence customer churn.
* Long-term contracts improve customer retention.
---
## Data Preprocessing
* Converted TotalCharges to numeric format.
* Handled missing values using median imputation.
* Removed unnecessary columns such as CustomerID.
* Applied One-Hot Encoding for categorical variables.
* Encoded target variable using Label Encoding.
* Split data into training and testing datasets.
---
## Machine Learning Models
### Logistic Regression
* Accuracy: **82.11%**
### Random Forest Classifier
* Accuracy: **78.92%**
--
## Model Evaluation
Evaluation Metrics:
* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
Logistic Regression achieved the best performance on this dataset.
---
## Technologies Used
* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Joblib
* Jupyter Notebook
* Streamlit
---
## Project Structure

```text
Customer-Churn-Prediction/
│
├── app.py
├── README.md
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── notebooks/
    ├── customer_churn_EDA.ipynb
    ├── Customer_Churn_Modeling.ipynb
    └── churn_model.pkl
```
---

## How to Run
### Clone Repository
```bash
git clone https://github.com/Suman-Hazra-12/customer-churn-prediction
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Run Streamlit App
```bash
streamlit run app.py
```
---
## Future Improvements
* Hyperparameter tuning
* XGBoost implementation
* Customer risk scoring
* Real-time prediction API
* Cloud deployment
* Interactive business dashboard
