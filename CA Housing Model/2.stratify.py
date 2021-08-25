import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 50)

data = pd.read_csv('housing.csv')

# split into training and test data sets
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

###########################
# let's stratify by income
###########################

# create bucket with the pandas cut function
data['income_cat'] = pd.cut(data['median_income'],
 bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
 labels=[1, 2, 3, 4, 5])

# show the hist of the bucketed field
data['income_cat'].hist()
plt.show()

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data['income_cat']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

# test stratified sample (can similarly check full data bucket and unstratified)
strat_test_set['income_cat'].value_counts() / len(strat_test_set)

# drop income_cat from training sets
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)

