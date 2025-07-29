import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

def prepare_data():
    data = pd.read_csv("data/CCA_data.csv")
    data = data.dropna()

    X = data.drop('Churn', axis=1)
    Y = data['Churn'].map({'Yes':1, 'No':0})

    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64','float64']).columns

    return X, Y, categorical_cols, numerical_cols
