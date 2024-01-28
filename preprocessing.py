import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.impute import KNNImputer

mci = pd.read_csv('data/mci_preprocessed_wo_csf.csv')
codes = {'CN_MCI': 0, 'Dementia': 1}
mci['last_DX'].replace(codes, inplace=True)
mci = mci.drop(['Unnamed: 0'], axis=1)

mci.dtypes

# find number of nan values for each column and order by most to least
mci.isna().sum().sort_values(ascending=False)

# use knn imputer to fill in nan values


imputer = KNNImputer(n_neighbors=5)
mci.iloc[:,1:] = imputer.fit_transform(mci.iloc[:,1:])

# find count of unique values for each last_DX
mci['last_DX'].value_counts()

# use SMOTE to oversample minority class

X = mci.iloc[:,1:]
y = mci.iloc[:,0]
print('Original dataset shape %s' % Counter(y))
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))

# create dataframe from X_res and y_res with last_DX as first column
mci = pd.DataFrame(X_res)
mci.insert(0, 'last_DX', y_res)

# convert to float32
mci.iloc[:,1:] = mci.iloc[:,1:].astype('float32')

# scale data
scaler = StandardScaler()
x = scaler.fit_transform(mci.iloc[:,1:])
combined = np.concatenate((mci.iloc[:,:1], x), axis=1)

# restore column names and save to csv
combined = pd.DataFrame(combined)
combined.columns = mci.columns
combined.to_csv('data/mci_preprocessed_wo_csf_vae.csv', index=False)

# preprocess data for real data
mci = pd.read_csv('data/mci_preprocessed_wo_csf.csv')
codes = {'CN_MCI': 0, 'Dementia': 1}
mci['last_DX'].replace(codes, inplace=True)
mci = mci.drop(['Unnamed: 0'], axis=1)

mci.dtypes

# find number of nan values for each column and order by most to least
mci.isna().sum().sort_values(ascending=False)

# use knn imputer to fill in nan values
imputer = KNNImputer(n_neighbors=5)
mci.iloc[:,2:] = imputer.fit_transform(mci.iloc[:,2:])

# find count of unique values for each last_DX
mci['last_DX'].value_counts()

# use SMOTE to oversample minority class
X = mci.iloc[:,1:]
y = mci.iloc[:,0]
print('Original dataset shape %s' % Counter(y))
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))

# create dataframe from X_res and y_res with last_DX as first column
mci = pd.DataFrame(X_res)
mci.insert(0, 'last_DX', y_res)

# convert to float32
mci.iloc[:,2:] = mci.iloc[:,2:].astype('float32')

scaler = StandardScaler()
x = scaler.fit_transform(mci.iloc[:,2:])
combined = np.concatenate((mci.iloc[:,:2], x), axis=1)

# restore column names and save to csv
combined = pd.DataFrame(combined)
combined.columns = mci.columns
combined.to_csv('data/mci_preprocessed_wo_csf_real.csv', index=False)