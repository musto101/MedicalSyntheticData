# run survival random forest model on synthetic data for CN

# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import GridSearchCV
from lifelines.utils import concordance_index

# read in data
data = pd.read_csv('data/generated_mci_data_clean2.csv')

# change last_DX to boolean
data['last_DX'] = data['last_DX'].astype(bool)

# change last_visit to absolute value
# data['last_visit'] = data['last_visit'].abs()

# print counts of last_DX
print(data['last_DX'].value_counts())

# print summary statistics for last_visit
print(data['last_visit'].describe())

# split data into train and test sets
train, test = train_test_split(data, test_size=0.2, random_state=0)

# create validation set
train, val = train_test_split(train, test_size=0.2, random_state=0)

# split train, val, and test sets into X and y with y containing last_DX and last_visit
y_train = train[['last_DX', 'last_visit']]
y_train = y_train.to_records(index=False)
train = train.drop(['last_DX', 'last_visit'], axis=1)
y_val = val[['last_DX', 'last_visit']].to_records(index=False)
val = val.drop(['last_DX', 'last_visit'], axis=1)
y_test = test[['last_DX', 'last_visit']]
test = test.drop(['last_DX', 'last_visit'], axis=1)

# define parameter grid
params = {
    'n_estimators': [50, 100, 500, 1000, 1500, 2000],
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'min_samples_split': [2, 4, 6, 8, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'max_features': ['auto', 'sqrt', 'log2']
}

# define grid search
grid_search = GridSearchCV(estimator=RandomSurvivalForest(n_jobs=-1), param_grid=params, cv=5, verbose=1, n_jobs=-1)

# fit grid search
grid_search.fit(train, y_train)

# print best parameters
print(grid_search.best_params_)
# print best score
print(grid_search.best_score_)

# use best model to predict on test set
y_pred = grid_search.predict(test) # 0.85

# round predictions to nearest integer
y_pred = np.round(y_pred)

# calculate concordance index for test set
concordance_index(y_test['last_visit'], -y_pred)

