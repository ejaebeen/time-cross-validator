import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.model_selection import TimeSeriesSplit  
from sklearn.metrics import accuracy_score

def obtain_fraud_lag_applied_index(
    y: pd.Series, 
    time_ref: pd.Series, 
    fraud_identified_time_ref: pd.Series, 
    train_fraud_lag_until: datetime, 
    train_fold_days_until: datetime,
    test_fraud_lag_until: datetime,
    test_fold_days_until: datetime,
):
    """
    apply logic on fraud lag filtering to identify what data to select for train/test
    for train select:
        transactions before the fraud lag OR
        fraud transactions identified as fraud before the end of the day for the fold
    for test select:
        transactions not selected in train AND (
            transactions before the fraud lag OR
            fraud transactions identified as fraud before the end of the day for the fold
        )
    """          
    train_idx = time_ref.loc[
        (time_ref <= train_fraud_lag_until) | 
        (y & (fraud_identified_time_ref <= train_fold_days_until))
    ].index
    test_idx = time_ref.loc[
        (~time_ref.index.isin(train_idx)) &
        (
            (time_ref <= test_fraud_lag_until) | 
            (y & (fraud_identified_time_ref <= test_fold_days_until))
        )
    ].index

    return train_idx, test_idx



class FraudTimeSplit:
    """
    fraud time splitter. It returns the iterables containing index of
    the train and test data which can be passed on as cv in 
    sklearn.model_selection.cross_validate
    Similar structure to sklearn.model_selection.TimeSeriesSplit  
    """
    def __init__(
        self,
        n_splits: int = 5,
        fraud_lag: int = 90
    ):
        self.n_splits = n_splits
        self.fraud_lag = fraud_lag

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_ref: pd.Series,
        fraud_identified_time_ref: pd.Series,
    ):
        if not pd.api.types.is_datetime64_ns_dtype(time_ref):
            raise ValueError("time_ref must have datetime64_ns dtype")
        if not pd.api.types.is_datetime64_ns_dtype(fraud_identified_time_ref):
            raise ValueError("fraud_identified_time_ref must have datetime64_ns dtype")
        if not (len(X) == len(y) == len(time_ref) == len(fraud_identified_time_ref)):
            raise ValueError("all inputs must have same length")
        if not (y.index == time_ref.index).all():
            raise ValueError("all inputs must have same index")
        if not (y.index == fraud_identified_time_ref.index).all():
            raise ValueError("all inputs must have same index")
        if not (time_ref.values == time_ref.sort_values().values).all():
            raise ValueError("time ref must be in ascending order")
            
        # apply normal time series split
        tscv = TimeSeriesSplit(self.n_splits)

        train_index_list = []
        test_index_list = []
        
        for train_index, test_index in tscv.split(time_ref):
            # obtain date until for each fold
            train_fold_days_until = time_ref.iloc[train_index].max()
            test_fold_days_until = time_ref.iloc[test_index].max()

            # obtain fraud lag date until for each fold
            train_fraud_lag_until = train_fold_days_until - timedelta(days=self.fraud_lag)
            test_fraud_lag_until = test_fold_days_until - timedelta(days=self.fraud_lag)

            # obtain index
            train_idx, test_idx = obtain_fraud_lag_applied_index(
                y, 
                time_ref, 
                fraud_identified_time_ref, 
                train_fraud_lag_until, 
                train_fold_days_until,
                test_fraud_lag_until,
                test_fold_days_until,
            )

            train_index_list.append(train_idx)
            test_index_list.append(test_idx)
        
        return zip(train_index_list, test_index_list)





