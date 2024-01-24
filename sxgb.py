# run survival xgboost model on synth data
import pandas as pd
import xgboost as xgb
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold

data = pd.read_csv('data/generated_cn_data_clean2.csv')

test = data.sample(frac=0.2, random_state=42)
train = data.drop(test.index)

# Splitting the data into features and target
X = train.drop(['last_DX', 'last_visit'], axis=1)
y = train[['last_visit', 'last_DX']]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

params = {
    'objective': "survival:cox",
    'eval_metric': "cox-nloglik",
    'booster': "gbtree",
    'nthread': -1,
    'num_boost_round': [50, 100],
    'max_depth': [1, 5, 10],
    'learning_rate': [0.001, 0.01],
    'eta': [0.01, 0.1],
    'min_child_weight': [0.0001, 0.001],
    'alpha': [0.001, 0.01],  # Changed from tuple to list
    'gamma': [0, 0.1, 0.2]
}

c_indices = []
best_params = {}
for num_boost_round in params['num_boost_round']:
    print(f"n_estimators: {num_boost_round}")
    for max_depth in params['max_depth']:
        print(f"max_depth: {max_depth}")
        for learning_rate in params['learning_rate']:
            print(f"learning_rate: {learning_rate}")
            for eta in params['eta']:
                print(f"eta: {eta}")
                for min_child_weight in params['min_child_weight']:
                    print(f"min_child_weight: {min_child_weight}")
                    for alpha in params['alpha']:
                        print(f"alpha: {alpha}")
                        for gamma in params['gamma']:
                            print(f"gamma: {gamma}")
                            current_params = {
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
                            for train_idx, test_idx in kf.split(X):
                                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                                dtrain = xgb.DMatrix(X_train, label=y_train['last_visit'], weight=y_train['last_DX'])
                                dtest = xgb.DMatrix(X_test, label=y_test['last_visit'], weight=y_test['last_DX'])

                                # print(f"Training fold {train_idx + 1}")
                                bst = xgb.train(current_params, dtrain, early_stopping_rounds=5, evals=[(dtest, 'eval')], )
                                print(f"Best score: {bst.best_score}")
                                # best_params.update(bst.best_params)

                                # calculate concordance index for test set
                                c_index = concordance_index(y_test['last_visit'], -bst.predict(dtest))
                                c_indices.append(c_index)
                                print(f"Concordance index: {c_index}")

                                # save best parameters
                                if c_index == max(c_indices):
                                    best_params = current_params


print(f"Best parameters: {best_params}")

# run survival xgboost model with best parameters
dtrain = xgb.DMatrix(X, label=y['last_visit'], weight=y['last_DX'])
dtest = xgb.DMatrix(test.drop(['last_DX', 'last_visit'], axis=1), label=test['last_visit'], weight=test['last_DX'])


final_model = xgb.train(best_params, dtrain)

y_test = test[['last_visit', 'last_DX']]
# calculate concordance index for test set
c_index = concordance_index(y_test['last_visit'], -final_model.predict(dtest))
print(f"Test Concordance index: {c_index}")
