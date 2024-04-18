import pytest
from ftcv.fraud_time_cross_validate import fraud_time_cross_validate, relabel_fraud_detected_after_cv_time
import pandas as pd
from datetime import datetime
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

@pytest.fixture()
def fraud_df():
    # create synthetic data for testing - use example.csv
    df = pd.read_csv("example/example.csv", index_col=0)

    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['fraud_identified_date'] = pd.to_datetime(df['fraud_identified_date'])

    df = df.sort_values('transaction_date')
    df = df.reset_index(drop=True)

    X = df[['transaction_amount','1','2','3','4','5','6','7','8']]
    y = df['fraud']
    date_ref = df['transaction_date']
    fraud_detected_date_ref = df['fraud_identified_date']
    
    return {
        'X':X, 
        'y':y, 
        'time_ref':date_ref, 
        'fraud_identified_time_ref':fraud_detected_date_ref
    }

def test_relabel_fraud_detected_after_cv_time():
    y = pd.Series([True, True, True, True, True, True, False, True, False, True])
    fraud_identified_time_ref = pd.Series([
        datetime(2020,1,5),
        datetime(2020,1,5),
        datetime(2020,2,21),
        datetime(2020,2,21),
        datetime(2020,3,21),
        datetime(2020,3,21),
        None,
        datetime(2020,5,20),
        None,
        datetime(2020,5,7)
    ])

    y_relabel = relabel_fraud_detected_after_cv_time(
        y, 
        fraud_identified_time_ref,
        fold_days_until=datetime(2020,5,6)
    )
    assert (y_relabel == [True, True, True, True, True, True, False, False, False, False]).all()

def test_fraud_time_cross_validate_sklearn_pipeline(fraud_df):
    # test to see if it works on sklearn.pipeline.Pipeline
    pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression())])
    fraud_time_cross_validate(
        estimator=pipe,
        n_splits=2,
        **fraud_df
    )

def test_fraud_time_cross_validate_xgb(fraud_df):
    # test to see if it works on XGBClassifier
    xgb = XGBClassifier()
    fraud_time_cross_validate(
        estimator=xgb,
        n_splits=2,
        **fraud_df
    )

def test_fraud_time_cross_validate_sklearn_model(fraud_df):
    # test to see if it works on sklearn model
    lr = LogisticRegression()
    fraud_time_cross_validate(
        estimator=lr,
        n_splits=2,
        **fraud_df
    )