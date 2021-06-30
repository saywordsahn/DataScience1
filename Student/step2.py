import pandas as pd
pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 50)
file_path = 'train.csv'

data = pd.read_csv(file_path)

# DO: have a look through the columns to find what it's called

# SOL:
# print(data.columns)

# DO: define our prediction target
# DO: define our feature names (give to student)
# DO: define our predictors (x value)
y = data.SalePrice

feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

x = data[feature_names]

# DO: let's check to make sure our predictor data looks reasonable
# DO: check the first few rows of our data

# SOL:
# print(x.describe())
# print(x.head())

# DO: create our model (we'll need an import statement)
# DO: make sure we set random_state = 1
# DO: fit the model
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state=1)

model.fit(x, y)

# DO: make our predictions for the data set and output them
predictions = model.predict(x)
print(predictions)

# DO: check our results from our predictions

# SOL:
print(y.head(30))
print(model.predict(x.head(30)))



# Get the MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y, predictions)
print(mae)



