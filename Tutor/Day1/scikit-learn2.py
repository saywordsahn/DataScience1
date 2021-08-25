import pandas as pd
from sklearn.tree import DecisionTreeRegressor

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

# model validation!

# note: let's update our feature names:
# ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
#                         'YearBuilt', 'Lattitude', 'Longtitude']

# THINK: how can we test for how well our model prices?
# Simple way is Mean Absolute Error (MAE)
# error = actual - predicted
# MAE = sum of absolute values of errors / n

from sklearn.metrics import mean_absolute_error

predicted_home_prices = model.predict(x)
mae = mean_absolute_error(y, predicted_home_prices)

# print(mae)

# THINK: Is it a good idea to use the data we trained our model with to
# test our model for predictive accuracy?

# What we did is called an in-sample score, because we tested our model against
# the data we trained it with

# Imagine that, in the large real estate market, door color is unrelated to home price.
#
# However, in the sample of data you used to build the model,
# all homes with green doors were very expensive.
#
# The model's job is to find patterns that predict home prices,
# so it will see this pattern,
# and it will always predict high prices for homes with green doors.


# we need a seperate set of data called validation data to test model performance

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we

train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=0)
# Define model
model = DecisionTreeRegressor(random_state=1)
# Fit model
model.fit(train_x, train_y)

# get predicted prices on validation data
val_predictions = model.predict(val_x)
print(mean_absolute_error(val_y, val_predictions))

# THINK: the difference is huge! Our out-of-sample prediction is over $250,000
# our in-sample data was around $400






