import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 50)
file_path = 'train.csv'
data = pd.read_csv(file_path)
y = data.SalePrice
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
x = data[feature_names]
model = DecisionTreeRegressor(random_state=1)
model.fit(x, y)
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=1)
model.fit(train_x, train_y)
val_predictions = model.predict(val_x)

mae = mean_absolute_error(val_y, val_predictions)
print(mae)


# DO: create function get the best tree size from max_leafs = [5, 25, 50, 100, 250, 500]


# create final model: now that we know the best tree size,
# we can create our final model using all of the data
# (not just the test sample)



