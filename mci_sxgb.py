# run survival xgboost model on synth data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('data/generated_mci_data_clean2.csv')

# prin counts of last_DX
print(data['last_DX'].value_counts())

# split data into train and test sets
train, test = train_test_split(data, test_size=0.2, random_state=0)

# create validation set
train, val = train_test_split(train, test_size=0.2, random_state=0)

# split train, val, and test sets into X and y with y containing last_DX and last_visit
y_train = train[['last_DX', 'last_visit']]
train = train.drop(['last_DX', 'last_visit'], axis=1)
y_val = val[['last_DX', 'last_visit']]
val = val.drop(['last_DX', 'last_visit'], axis=1)
y_test = test[['last_DX', 'last_visit']]
test = test.drop(['last_DX', 'last_visit'], axis=1)

# run survival xgboost model
xgb_model = XGBClassifier(objective="survival:cox", random_state=0)

# define parameters
params = {
    'objective': ["survival:cox"],
    'eval_metric': ["cox-nloglik"],
    'booster': ["gbtree"],
    'nthread': [-1],
    'n_estimators': [5, 10, 20, 50, 100],
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'learning_rate': [0.001, 0.01],
    'eta': [0.01, 0.1],
    'min_child_weight': [0.0001, 0.001, 0.01],
    'alpha': [0.001, 0.01, 0.1],
    'gamma': [0, 0.1, 0.2, 1]
}

# define grid search
grid_search = GridSearchCV(estimator=xgb_model, param_grid=params, cv=5, verbose=1, n_jobs=-1)

# fit grid search with early stopping
grid_search.fit(train, y_train, early_stopping_rounds=5, eval_set=[(val, y_val)])

# print best parameters
print(grid_search.best_params_)

# print best score
print(grid_search.best_score_)

