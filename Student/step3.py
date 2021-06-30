import pandas as pd
from sklearn.tree import DecisionTreeRegressor

pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 50)
file_path = 'train.csv'
data = pd.read_csv(file_path)
y = data.SalePrice
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
x = data[feature_names]
model = DecisionTreeRegressor(random_state=1)
model.fit(x, y)

# DO: split our data into training and validation variables
from sklearn.model_selection import train_test_split

train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=1)



# DO: fit the model with the training data
# DO: Make predictions with validation data
# DO: print the top few prices from the validation data
# DO: print the top we predicted prices from the validation data
model.fit(train_x, train_y)

val_predictions = model.predict(val_x)

# SOL: print(model.predict(val_x.head(10)))
# SOL: print(val_y.head(10))

# THINK: what's different from when we tested against our in-sample?



# DO: calculate the Mean Absolute Error (MAE)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(val_y, val_predictions)

print(mae)





