# run survival xgboost model on synth data
import xgboost as xgb
from lifelines.utils import concordance_index
import pyswarms as ps
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


data = pd.read_csv('data/generated_cn_data_clean2.csv')
real_data = pd.read_csv('data/cn_preprocessed_wo_csf_real.csv')

val, test = train_test_split(real_data, test_size=0.5, random_state=0)
train = data

# Splitting the data into features and target
X = train.drop(['last_DX', 'last_visit'], axis=1)
y = train[['last_visit', 'last_DX']]

X_val = val.drop(['last_DX', 'last_visit'], axis=1)
y_val = val[['last_visit', 'last_DX']]

dtrain = xgb.DMatrix(X, label=y['last_visit'], weight=y['last_DX'])
dval = xgb.DMatrix(X_val, label=y_val['last_visit'], weight=y_val['last_DX'])

# create a PSO for hyperparameter tuning on the survival xgboost model

lower_bound = [50, 1, 0.001, 0.01, 0.0001, 0.001, 0]
upper_bound = [100, 10, 0.01, 0.1, 0.001, 0.01, 0.2]

bounds = (lower_bound, upper_bound)

# define objective function
def objective_function(x):

    num_boost_round = x[0]
    # make sure parameters are integers
    num_boost_round = int(round(num_boost_round,0))
    max_depth = x[1]
    # make sure parameters are integers
    max_depth = int(round(max_depth,0))
    learning_rate = x[2]

    eta = x[3]
    min_child_weight = x[4]
    alpha = x[5]
    gamma = x[6]

    params = {
        'objective': "survival:cox",
        'eval_metric': "cox-nloglik",
        'booster': "gbtree",
        'nthread': -1,
        'num_boost_round': num_boost_round,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'eta': eta,
        'min_child_weight': min_child_weight,
        'alpha': alpha,
        'gamma': gamma
    }

    # create survival xgboost model
    model = xgb.train(params, dtrain, early_stopping_rounds=5, evals=[(dval, 'eval')])
    # fit model

    # predict on validation set
    y_pred = model.predict(dval)
    # round predictions to nearest integer
    y_pred = np.round(y_pred)

    # calculate negative concordance index for validation set
    c_index = concordance_index(y_val['last_visit'], -y_pred)

    return -c_index

# define enforce hyperparameters constraint function
def enforce_hyperparameter_constraints(particle):
    min_num_boost_round, max_num_boost_round = 50, 100
    min_max_depth, max_max_depth = 1, 10
    min_learning_rate, max_learning_rate = 0.001, 0.01
    min_eta, max_eta = 0.01, 0.1
    min_min_child_weight, max_min_child_weight = 0.0001, 0.001
    min_alpha, max_alpha = 0.001, 0.01
    min_gamma, max_gamma = 0, 0.2

    if particle[0] < min_num_boost_round:
        particle[0] = min_num_boost_round
    elif particle[0] > max_num_boost_round:
        particle[0] = max_num_boost_round

    if particle[1] < min_max_depth:
        particle[1] = min_max_depth
    elif particle[1] > max_max_depth:
        particle[1] = max_max_depth

    if particle[2] < min_learning_rate:
        particle[2] = min_learning_rate
    elif particle[2] > max_learning_rate:
        particle[2] = max_learning_rate

    if particle[3] < min_eta:
        particle[3] = min_eta
    elif particle[3] > max_eta:
        particle[3] = max_eta

    if particle[4] < min_min_child_weight:
        particle[4] = min_min_child_weight
    elif particle[4] > max_min_child_weight:
        particle[4] = max_min_child_weight

    if particle[5] < min_alpha:
        particle[5] = min_alpha
    elif particle[5] > max_alpha:
        particle[5] = max_alpha

    if particle[6] < min_gamma:
        particle[6] = min_gamma
    elif particle[6] > max_gamma:
        particle[6] = max_gamma

    # enforce the correct type for each element of particle
    particle[0] = int(round(particle[0],0))
    particle[1] = int(round(particle[1],0))
    particle[2] = float(particle[2])
    particle[3] = float(particle[3])
    particle[4] = float(particle[4])
    particle[5] = float(particle[5])
    particle[6] = float(particle[6])

    # print(particle)
    return particle


# define constrained objective function
def constrained_objective_function(particles, *args, **kwargs):
    # Adjust each particle
    for i in range(particles.shape[0]):
        particles[i, :] = enforce_hyperparameter_constraints(particles[i, :])

    # Evaluate the objective function for all particles
    fitness = np.apply_along_axis(objective_function, 1, particles)
    return fitness


options = {'c1': 2, 'c2': 2, 'w':0.9} # define hyperparameters for PSO

# define optimizer
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=7, options=options, bounds=bounds)

# perform optimization
cost, pos = optimizer.optimize(constrained_objective_function, iters=100, verbose=True)

# run survival xgboost model with best parameters

num_boost_round = pos[0]
# make sure parameters are integers
num_boost_round = int(round(num_boost_round,0))
max_depth = pos[1]
# make sure parameters are integers
max_depth = int(round(max_depth,0))
learning_rate = pos[2]
eta = pos[3]
min_child_weight = pos[4]
alpha = pos[5]
gamma = pos[6]

params = {
    'objective': "survival:cox",
    'eval_metric': "cox-nloglik",
    'booster': "gbtree",
    'nthread': -1,
    'num_boost_round': num_boost_round,
    'max_depth': max_depth,
    'learning_rate': learning_rate,
    'eta': eta,
    'min_child_weight': min_child_weight,
    'alpha': alpha,
    'gamma': gamma
}

# create survival xgboost model
model = xgb.train(params, dtrain, early_stopping_rounds=5, evals=[(dval, 'eval')])

# predict on test set
dtest = xgb.DMatrix(test.drop(['last_DX', 'last_visit'], axis=1), label=test['last_visit'], weight=test['last_DX'])
y_pred = model.predict(dtest)
# round predictions to nearest integer
y_pred = np.round(y_pred)

# calculate concordance index for test set
c_index = concordance_index(test['last_visit'], -y_pred)
