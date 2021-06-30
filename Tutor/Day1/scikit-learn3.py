import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 50)
file_path = 'melb_data.csv'
data = pd.read_csv(file_path)
data = data.dropna(axis=0)
y = data.Price
features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                        'YearBuilt', 'Lattitude', 'Longtitude']
x = data[features]
model = DecisionTreeRegressor(random_state=1)
model.fit(x, y)
predicted_home_prices = model.predict(x)
mae = mean_absolute_error(y, predicted_home_prices)
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=0)
model = DecisionTreeRegressor(random_state=1)
model.fit(train_x, train_y)
val_predictions = model.predict(val_x)
print(mean_absolute_error(val_y, val_predictions))



# Let's play around with how many leaf-nodes are generated

def get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_x, train_y)
    preds_val = model.predict(val_x)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# THINK: which is the optimal number of nodes?