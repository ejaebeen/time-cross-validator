from sklearn.model_selection import cross_validate
from ftcv.FraudTimeSplit import FraudTimeSplit
from sklearn.metrics import accuracy_score
from datetime import timedelta

def relabel_fraud_detected_after_cv_time(
    y, 
    fraud_identified_time_ref,
    fold_days_until
):
    y.loc[
        (fraud_identified_time_ref > fold_days_until) & 
        (y)
    ] = False

    return y

def fraud_time_cross_validate(
    estimator,
    X,
    y,
    time_ref,
    fraud_identified_time_ref,
    n_splits,
    fraud_lag=90
):
    """
    cross validator using fraud time split. 
    Fraud cases that are detected after the split time date but with transaction date before the fraud time lag are 
    changed to False i.e. wrong label
    """
    ftscv = FraudTimeSplit(n_splits, fraud_lag)
    cv = ftscv.split(X, y, time_ref, fraud_identified_time_ref)
    score = []
    for train_idx, test_idx in cv:
        train_X, test_X = X.loc[train_idx], X.loc[test_idx]
        train_y, test_y = y.loc[train_idx], y.loc[test_idx] 
        train_fraud_identified_time_ref = fraud_identified_time_ref.loc[train_idx]

        train_fold_days_until = time_ref.loc[train_idx].max()
        # relabel
        train_y_relabelled = relabel_fraud_detected_after_cv_time(
            train_y, 
            train_fraud_identified_time_ref, 
            train_fold_days_until
        )
        
        estimator.fit(train_X, train_y_relabelled)
        pred_y = estimator.predict(test_X)
        score.append(accuracy_score(test_y, pred_y))

    return score