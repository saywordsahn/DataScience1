import pandas as pd

pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 50)
file_path = 'melb_data.csv'
data = pd.read_csv(file_path)
data = data.dropna(axis=0)
y = data.Price
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x = data[features]

# Define: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
# Fit: Capture patterns from provided data. This is the heart of modeling.
# Predict: Just what it sounds like
# Evaluate: Determine how accurate the model's predictions are.

# import decision tree regressor from sklearn library
# why don't we use DecisionTreeClassifier?

from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
model = DecisionTreeRegressor(random_state=1)

# Fit model
model.fit(x, y)

print("Making predictions for the following 5 houses:")
print(x.head())
print("The predictions are:")
print(model.predict(x.head()))




