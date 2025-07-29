from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import logging
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.compose import ColumnTransformer
from data import prepare_data
import pandas as pd
import numpy as np
import joblib
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(X=None, Y=None):
    try:
        if X is None or Y is None:
            X, Y, categorical_cols, numerical_cols = prepare_data()

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop='first'), categorical_cols)
            ]
        )

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier())
        ])

        param_distributions = {
            'classifier__n_estimators': [100, 200, 300, 400, 500],
            'classifier__max_depth': [10, 20, 30, 40, 50, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['auto', 'sqrt'],
            'classifier__class_weight': [{0:1, 1:v} for v in np.linspace(1, 20, 30)]
        }

        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=20,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42
        )

        random_search.fit(X_train,Y_train)

        best_model = random_search.best_estimator_
        Y_pred = best_model.predict(X_test)

        results = {
            'best_params': random_search.best_params_,
            'cv_score': random_search.best_score_,
            'test_score': roc_auc_score(Y_test, Y_pred),
            'classification_report': classification_report(Y_test, Y_pred)
        }

        return best_model, results

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise



def save_model(model, filename="best_model.pkl"):
    joblib.dump(model, filename)
    return filename

def make_prediction(data):
    try:
        try:
            model = joblib.load('best_model.pkl')
        except:
            model, _ = train_model()
            save_model(model)

        input_df = pd.DataFrame([data])

        _, _, categorical_cols, numerical_cols = prepare_data()

        for col in categorical_cols:
            if col not in input_df.columns:
                input_df[col] = None
        for col in numerical_cols:
            if col not in input_df.columns:
                input_df[col] = 0

        prediction = model.predict(input_df)[0]
        return prediction
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return 0