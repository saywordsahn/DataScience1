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


#############################
# Correlations
#############################


corr_matrix = data.corr()
print(corr_matrix)

print(corr_matrix["median_house_value"].sort_values(ascending=False))

# we could generate all correlations but there would be too many,
# let's just check a few that could be important
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
 "housing_median_age"]
scatter_matrix(data[attributes], figsize=(12, 8))
plt.show()

# let's zoom in on med income / med house value
data.plot(kind="scatter", x="median_income", y="median_house_value",
 alpha=0.1)
plt.show()


# let's try adding some modified fields
data["rooms_per_household"] = data["total_rooms"]/data["households"]
data["bedrooms_per_room"] = data["total_bedrooms"]/data["total_rooms"]
data["population_per_household"]=data["population"]/data["households"]

corr_matrix = data.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))
