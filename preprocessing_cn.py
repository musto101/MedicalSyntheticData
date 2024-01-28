import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.impute import KNNImputer

cn = pd.read_csv('data/cn_preprocessed_wo_csf.csv')
codes = {'CN': 0, 'MCI_AD': 1}
cn['last_DX'].replace(codes, inplace=True)
cn = cn.drop(['Unnamed: 0'], axis=1)

cn.dtypes

# find number of nan values for each column and order by most to least
cn.isna().sum().sort_values(ascending=False)

# use knn imputer to fill in nan values
imputer = KNNImputer(n_neighbors=5)
cn.iloc[:,1:] = imputer.fit_transform(cn.iloc[:,1:])

# find count of unique values for each last_DX
cn['last_DX'].value_counts()

# use SMOTE to oversample minority class
X = cn.iloc[:,1:]
y = cn.iloc[:,0]
print('Original dataset shape %s' % Counter(y))
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))

# create dataframe from X_res and y_res with last_DX as first column
cn = pd.DataFrame(X_res)
cn.insert(0, 'last_DX', y_res)

# convert to float32
cn.iloc[:,1:] = cn.iloc[:,1:].astype('float32')

# scale data
scaler = StandardScaler()
x = scaler.fit_transform(cn.iloc[:,1:])
combined = np.concatenate((cn.iloc[:,:1], x), axis=1)

# restore column names and save to csv
combined = pd.DataFrame(combined)
combined.columns = cn.columns
combined.to_csv('data/cn_preprocessed_wo_csf_vae.csv', index=False)

# preprocess data for real data

cn = pd.read_csv('data/cn_preprocessed_wo_csf.csv')
codes = {'CN': 0, 'MCI_AD': 1}
cn['last_DX'].replace(codes, inplace=True)
cn = cn.drop(['Unnamed: 0'], axis=1)

cn.dtypes

# find number of nan values for each column and order by most to least
cn.isna().sum().sort_values(ascending=False)

# use knn imputer to fill in nan values
imputer = KNNImputer(n_neighbors=5)
cn.iloc[:,2:] = imputer.fit_transform(cn.iloc[:,2:])

# find count of unique values for each last_DX
cn['last_DX'].value_counts()

# use SMOTE to oversample minority class
X = cn.iloc[:,1:]
y = cn.iloc[:,0]
print('Original dataset shape %s' % Counter(y))
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))

# create dataframe from X_res and y_res with last_DX as first column
cn = pd.DataFrame(X_res)
cn.insert(0, 'last_DX', y_res)

# convert to float32
cn.iloc[:,2:] = cn.iloc[:,2:].astype('float32')

scaler = StandardScaler()
x = scaler.fit_transform(cn.iloc[:,2:])
combined = np.concatenate((cn.iloc[:,:2], x), axis=1)

# restore column names and save to csv
combined = pd.DataFrame(combined)
combined.columns = cn.columns
combined.to_csv('data/cn_preprocessed_wo_csf_real.csv', index=False)