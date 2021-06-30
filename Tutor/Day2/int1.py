import pandas as pd
from sklearn.model_selection import train_test_split
pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 50)

# Read the data
x_full = pd.read_csv('train.csv', index_col='Id')
X_test_full = pd.read_csv('test.csv', index_col='Id')

# Obtain target and predictors
y = x_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
x = x_full[features].copy()
x_test = X_test_full[features].copy()

# Break off validation set from training data
x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)
print(x_train.head(15))








from sklearn.ensemble import RandomForestRegressor

# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]






from sklearn.metrics import mean_absolute_error


# Function for comparing different models
def score_model(model, x_t=x_train, x_v=x_valid, y_t=y_train, y_v=y_valid):
    model.fit(x_t, y_t)
    preds = model.predict(x_v)
    return mean_absolute_error(y_v, preds)


for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))
