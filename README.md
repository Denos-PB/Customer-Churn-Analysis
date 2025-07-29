# Customer-Churn-Analysis
Web-based application that predicts whether a customer is likely to churn (stop using a service) based on historical data.

## Overview
This project implements a machine learning model to predict customer churn using historical customer data. The application provides a web interface for training the model and making predictions.

## Features
- Data preparation and preprocessing
- Machine learning model training with hyperparameter tuning
- Web interface for making predictions
- Prediction history tracking

## Technical Details
- **ML Model**: Random Forest Classifier with hyperparameter optimization
- **Data Processing**: Automated handling of categorical and numerical features
- **Hyperparameter Tuning**: RandomizedSearchCV for efficient parameter search
- **Evaluation Metrics**: ROC-AUC score and classification report
- **Web Framework**: Django

## Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run migrations: `python manage.py migrate`
4. Start the server: `python manage.py runserver`

## Usage
### Training the Model
Navigate to the training page and click "Train Model" to train a new model with the latest data.

### Making Predictions
Fill in the customer information form with:
- Contract type
- Monthly charges
- Total charges
- Tenure (months)

The system will predict whether the customer is likely to churn.

### Viewing History
The history page shows all previous predictions made by the system.

## Data
The model uses customer data from `data/CCA_data.csv` which includes various customer attributes such as demographics, service usage, and contract details.

## Model Performance
The model is evaluated using ROC-AUC score and a detailed classification report including precision, recall, and F1-score metrics.
