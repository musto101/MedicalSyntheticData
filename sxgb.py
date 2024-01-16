# run survival xgboost model on synth data
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from lifelines.utils import to_long_format, add_covariate_to_timeline

data = pd.read_csv('data/generated_cn_data_clean2.csv')

# prin counts of last_DX
print(data['last_DX'].value_counts())

X = data.drop(columns=['last_DX', 'last_visit'])
y = data[['last_DX', 'last_visit']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform the dataset for survival analysis
train = to_long_format(X_train, duration_col='last_visit', event_col='last_DX')
test = to_long_format(X_test, duration_col='last_visit', event_col='last_DX')

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



# print best parameters
print(grid_search.best_params_)

# print best score
print(grid_search.best_score_)

