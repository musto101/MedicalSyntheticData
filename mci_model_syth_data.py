import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter

data = pd.read_csv('data/generated_mci_data2.csv')

print(data.head())

# data = data.drop(['Unnamed: 0'], axis=1)

# print summary statistics for last_DX
print(data['last_DX'].describe())
# print value counts for last_DX
print(data['last_DX'].value_counts())

# plot histogram of last_DX
plt.hist(data['last_DX'])
plt.show()

# change last_DX to 0 and 1 based on median with values lower than median = 0 and values higher than median = 1
median = data['last_DX'].median()
data['last_DX'] = np.where(data['last_DX'] <= median, 0, 1)

# print value counts for last_DX
print(data['last_DX'].value_counts())
# Find the minimum and maximum values in the 'last_visit' column
min_value = data['last_visit'].min()
max_value = data['last_visit'].max()

# Normalize the 'last_visit' column to a scale of 0 to the range (max - min)
# This will set the lowest value to 0 and the highest to (max - min)
data['normalized_last_visit'] = ((data['last_visit'] - min_value) / (max_value - min_value))

# Scale this to a range of integers
# Assuming the range represents a number of months between 6 and a maximum value
# Adjust the scale as needed based on the actual range of the study
min_months = 6  # Minimum months since baseline
max_months = 60  # Example maximum number of months
range_months = max_months - min_months
data['last_visit'] = (data['normalized_last_visit'] * range_months + min_months).round().astype(int)


# remove normalized_last_visit column
data = data.drop(['normalized_last_visit'], axis=1)

# print value counts for last_DX
print(data['last_DX'].value_counts())

# split data into train and test sets
train = data.sample(frac=0.8, random_state=0)
test = data.drop(train.index)

# run cox proportional hazard model
cph = CoxPHFitter()
cph.fit(train, duration_col='last_visit', event_col='last_DX')

# print summary of cox model
cph.print_summary()

# plot baseline survival curve
# cph.plot()
# plt.show()

# predict survival curve for test set
cph.predict_survival_function(test)

# calculate concordance index for test set
cph.score(test, scoring_method="concordance_index")

# save data to csv
data.to_csv('data/generated_mci_data_clean2.csv', index=False)
