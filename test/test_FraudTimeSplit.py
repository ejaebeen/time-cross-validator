import pytest
import ftcv.FraudTimeSplit
import pandas as pd
from datetime import datetime
import numpy as np

@pytest.fixture()
def fraud_df():
    # create synthetic data for testing
    # TODO: use simulator in synthetic-transaction-generator instead
    X = pd.DataFrame({
        'f1': [0,1,0,1,0,1,0,1,0,1],
        'f2': [1,100,1,100,1,100,1,99,1,100]
    })
    y = pd.Series([False, True, False, True, False, True, False, True, False, True])
    time_ref = pd.Series([
        datetime(2020,1,1),
        datetime(2020,1,2),
        datetime(2020,2,1),
        datetime(2020,2,1),
        datetime(2020,2,28),
        datetime(2020,3,2),
        datetime(2020,4,1),
        datetime(2020,4,2),
        datetime(2020,5,1),
        datetime(2020,5,5),
    ])
    fraud_identified_time_ref = pd.Series([
        None,
        datetime(2020,1,5),
        None,
        datetime(2020,2,21),
        None,
        datetime(2020,3,21),
        None,
        datetime(2020,5,20),
        None,
        datetime(2020,5,6)
    ])

    return {
        'X':X, 
        'y':y, 
        'time_ref':time_ref, 
        'fraud_identified_time_ref':fraud_identified_time_ref
    }

def test_obtain_fraud_lag_applied_correct_output(fraud_df):
    # test that it returns the index as expected
    t_train_idx, t_test_idx = ftcv.FraudTimeSplit.obtain_fraud_lag_applied_index(
        y=fraud_df['y'],
        time_ref=fraud_df['time_ref'],
        fraud_identified_time_ref=fraud_df['fraud_identified_time_ref'],
        train_fraud_lag_until=datetime(2020,2,15), 
        train_fold_days_until=datetime(2020,3,3),
        test_fraud_lag_until=datetime(2020,4,18),
        test_fold_days_until=datetime(2020,5,7)
    )

    assert (t_train_idx.values == [0,1,2,3]).all()
    assert (t_test_idx.values == [4,5,6,7,9]).all()


def test_FraudTimeSplit_runs(fraud_df):
    # check that FraudTimeSplit runs without error
    ftcv_test = ftcv.FraudTimeSplit.FraudTimeSplit(fraud_lag=5, n_splits=2)
    ftcv_test.split(**fraud_df)


def test_FraudTimeSplit_different_length_returns_error(fraud_df):
    # check it returns error as expected
    ftcv_test = ftcv.FraudTimeSplit.FraudTimeSplit(fraud_lag=5, n_splits=2)
    expected_message=f"all inputs must have same length"
    with pytest.raises(ValueError, match=expected_message):
        fraud_df['fraud_identified_time_ref'] = fraud_df['fraud_identified_time_ref'].iloc[-1:]
        ftcv_test.split(**fraud_df)


def test_FraudTimeSplit_wrong_dtype_returns_error(fraud_df):
    # check it returns error as expected
    ftcv_test = ftcv.FraudTimeSplit.FraudTimeSplit(fraud_lag=5, n_splits=2)
    expected_message=f"time_ref must have datetime64_ns dtype"
    with pytest.raises(ValueError, match=expected_message):
        fraud_df['time_ref'] = pd.Series([0,1,2,3,4,5,6,7,8,9], dtype='int')
        ftcv_test.split(**fraud_df)


def test_FraudTimeSplit_wrong_date_order_returns_error(fraud_df):
    # check it returns error as expected
    ftcv_test = ftcv.FraudTimeSplit.FraudTimeSplit(fraud_lag=5, n_splits=2)
    expected_message=f"time ref must be in ascending order"
    with pytest.raises(ValueError, match=expected_message):
        fraud_df['time_ref'] = fraud_df['time_ref'].sample(random_state=121, frac=1).reset_index(drop=True)
        ftcv_test.split(**fraud_df)

# different index test