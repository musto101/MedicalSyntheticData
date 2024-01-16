# run a deephit model on the synthetic data for CN

# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pycox

# read in data
data = pd.read_csv('data/generated_cn_data_clean.csv')

# print counts of last_DX
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

# create parameter grid for deephit model
params = {
    'n_layers': [1, 2, 3],
    'n_units': [32, 64, 128],
    'batch_norm': [True, False],
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [0.01, 0.001, 0.0001],
    'batch_size': [64, 128, 256],
    'epochs': [50, 100, 150]
}

# create grid search
grid_search = DeepHitSingle.grid_search(train, y_train, val, y_val, params, metrics=['concordance_index'])


# run deephit model
model = DeepHitSingle()


