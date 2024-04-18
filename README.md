# fraud time cross validator

This is a mini-package to carry out the time cross validation with the fraud lag taken account into each fold. 

## How to run

please find the notebook that uses the ftcv in `example.ipynb`. Currently, it only supports accuracy score for the cross validation scores. You can set the number of days you want to apply the fraud lag using the param `fraud_lag`. 

For each fold, it works out what is the maximum date and goes back by fraud_lag days. Then:

- for train select:
    - transactions before the fraud lag OR
    - fraud transactions identified as fraud before the end of the day for the fold
- for train select:
    - transactions not selected in train AND (
        - transactions before the fraud lag OR
        - fraud transactions identified as fraud before the end of the day for the fold

Any fraud transaction that took place before the fraud lag date but identified as fraud after the end of the day for the fold, it is changed as non-fraud (i.e. wrong label) for TRAINING DATA ONLY (test does not change since we need to test on ground truth).  

To carry out unit tests, please run `python -m pytest`

## TODO
- More scores to be supported (precision, recall, f1, weighted accuracy, pr_auc, roc_auc etc.)
- rather than remove the non-fraud transactions happening after the fraud lag, we can put "uncertainty" to them by assigning lower weight to them in the training.
- logging to output information on each fold (date start, fraud lag date, number of transactions removed, number of fraud cases relabelled)
- sort out warning in pytest (coming from package dependency)