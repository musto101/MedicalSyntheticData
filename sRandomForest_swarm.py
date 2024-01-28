# run survival random forest model on synthetic data for CN

# import libraries
from lifelines.utils import concordance_index
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
import pyswarms as ps

# read in data
data = pd.read_csv('data/generated_cn_data_clean2.csv')

real_data = pd.read_csv('data/cn_preprocessed_wo_csf_real.csv')

val, test = train_test_split(real_data, test_size=0.5, random_state=0)

# change last_DX to boolean
data['last_DX'] = data['last_DX'].astype(bool)
val['last_DX'] = val['last_DX'].astype(bool)
test['last_DX'] = test['last_DX'].astype(bool)

train = data
# split train, val, and test sets into X and y with y containing last_DX and last_visit
y_train = train[['last_DX', 'last_visit']]
y_train = y_train.to_records(index=False)
train = train.drop(['last_DX', 'last_visit'], axis=1)
y_val = val[['last_DX', 'last_visit']].to_records(index=False)
val = val.drop(['last_DX', 'last_visit'], axis=1)
y_test = test[['last_DX', 'last_visit']]
test = test.drop(['last_DX', 'last_visit'], axis=1)

# read in data

lower_bound = [50, 1, 2, 1, 1]
upper_bound = [200, 9, 10, 5, 80]

bounds = (lower_bound, upper_bound)

# define objective function
def objective_function(x):
    # define parameters
    n_estimators = x[0]
    max_depth = x[1]
    min_samples_split = x[2]
    min_samples_leaf = x[3]
    max_features = x[4]
    # make sure parameters are integers
    n_estimators = int(round(n_estimators,0))
    max_depth = int(round(max_depth,0))
    min_samples_split = int(round(min_samples_split,0))
    min_samples_leaf = int(round(min_samples_leaf,0))
    max_features = int(round(max_features,0))

    # print(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features)
    # create survival random forest model
    rsf = RandomSurvivalForest(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                               min_samples_leaf=min_samples_leaf, max_features=max_features, n_jobs=-1)
    # fit model
    rsf.fit(train, y_train)
    # predict on test set
    y_pred = rsf.predict(val)
    # round predictions to nearest integer
    y_pred = np.round(y_pred)
    # calculate concordance index for test set
    c_index = concordance_index(y_val['last_visit'], -y_pred)
    return -c_index

# define enforce hyperparameters constraint function
def enforce_hyperparameter_constraints(particle):
    min_estimators, max_estimators = 50, 2000
    min_max_depth, max_max_depth = 1, 9
    min_min_samples_split, max_min_samples_split = 2, 10
    min_min_samples_leaf, max_min_samples_leaf = 1, 5
    min_max_features, max_max_features = 1, 80

    if particle[0] < min_estimators:
        particle[0] = min_estimators
    elif particle[0] > max_estimators:
        particle[0] = max_estimators

    if particle[1] < min_max_depth:
        particle[1] = min_max_depth
    elif particle[1] > max_max_depth:
        particle[1] = max_max_depth

    if particle[2] < min_min_samples_split:
        particle[2] = min_min_samples_split
    elif particle[2] > max_min_samples_split:
        particle[2] = max_min_samples_split

    if particle[3] < min_min_samples_leaf:
        particle[3] = min_min_samples_leaf
    elif particle[3] > max_min_samples_leaf:
        particle[3] = max_min_samples_leaf

    if particle[4] < min_max_features:
        particle[4] = min_max_features
    elif particle[4] > max_max_features:
        particle[4] = max_max_features

    # enforce integer constraint on each element of particle
    particle[0] = int(round(particle[0],0))
    particle[1] = int(round(particle[1],0))
    particle[2] = int(round(particle[2],0))
    particle[3] = int(round(particle[3],0))
    particle[4] = int(round(particle[4],0))

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
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=5, options=options, bounds=bounds)

# perform optimization
cost, pos = optimizer.optimize(constrained_objective_function, iters=100, verbose=True, n_processes=100)

# print best parameters
print(pos)
# round parameters to nearest integer
pos[0] = int(round(pos[0],0))
pos[1] = int(round(pos[1],0))
pos[2] = int(round(pos[2],0))
pos[3] = int(round(pos[3],0))
pos[4] = int(round(pos[4],0))

# print best cost
print(cost)

# use best model to predict on test set
rsf = RandomSurvivalForest(n_estimators=pos[0], max_depth=pos[1], min_samples_split=pos[2],
                            min_samples_leaf=pos[3], max_features=pos[4], n_jobs=-1)

# fit model
rsf.fit(train, y_train)

# predict on test set
y_pred = rsf.predict(test)

# round predictions to nearest integer
y_pred = np.round(y_pred)

# calculate concordance index for test set
concordance_index(y_test['last_visit'], -y_pred)







