import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 50)

data = pd.read_csv('housing.csv')

train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

data['income_cat'] = pd.cut(data['median_income'],
 bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
 labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data['income_cat']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)



###########################
# VISUALIZE
###########################

# copy training set so we don't mess it up
data = strat_train_set.copy()

# plot data by lat / long
data.plot(kind="scatter", x="longitude", y="latitude")
plt.show()

# use alpha property to get a better idea of density
data.plot(kind="scatter", x="longitude", y="latitude", alpha=.1)
plt.show()

data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
 s=data["population"]/100, label="population", figsize=(10,7),
 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()
plt.show()