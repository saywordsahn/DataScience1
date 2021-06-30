import pandas as pd
from sklearn.model_selection import train_test_split
pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 50)

# what to do about empty values in our data sets?

# 1. we can delete them, we may lose useful information if we do this
# 2. Imputation: fill in the missing value with something - this is often better than deleting
# 3. Imputation + field: we can imputate the missing value, and create a new variable that tells
#   our model if the value was imputated or not





# Load the data
data = pd.read_csv('melb_data.csv')

# Select target
y = data.Price

# To keep things simple, we'll use only numerical predictors
melb_predictors = data.drop(['Price'], axis=1)
x = melb_predictors.select_dtypes(exclude=['object'])

# Divide data into training and validation subsets
x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)



# define a function to measure the quality of each approach
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(x_train, x_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(x_train, y_train)
    preds = model.predict(x_valid)
    return mean_absolute_error(y_valid, preds)




# score the case where we just drop columns with missing values

# Get names of columns with missing values
cols_with_missing = [col for col in x_train.columns
                     if x_train[col].isnull().any()]

# be careful, we need to drop from the validation data sets too!
# Drop columns in training and validation data
reduced_X_train = x_train.drop(cols_with_missing, axis=1)
reduced_X_valid = x_valid.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))




# score with imputation

from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_x_train = pd.DataFrame(my_imputer.fit_transform(x_train))
imputed_x_valid = pd.DataFrame(my_imputer.transform(x_valid))

# Imputation removed column names; put them back
imputed_x_train.columns = x_train.columns
imputed_x_valid.columns = x_valid.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_x_train, imputed_x_valid, y_train, y_valid))

# print(imputed_x_train.describe())
# print(imputed_x_train.head(20))



# imputation and add isImputatedField variable
# Make copy to avoid changing original data (when imputing)
x_train_plus = x_train.copy()
x_valid_plus = x_valid.copy()

print('*************************')
print('original')
print('*************************')
print(x_train_plus.head(5))

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    x_train_plus[col + '_was_missing'] = x_train_plus[col].isnull()
    x_valid_plus[col + '_was_missing'] = x_valid_plus[col].isnull()

print('*************************')
print('add bool column for if data was missing')
print('*************************')
print(x_train_plus.head(5))

# Imputation
my_imputer = SimpleImputer()
imputed_x_train_plus = pd.DataFrame(my_imputer.fit_transform(x_train_plus))
imputed_x_valid_plus = pd.DataFrame(my_imputer.transform(x_valid_plus))


print('*************************')
print('imputated')
print('*************************')
print(imputed_x_train_plus.head(5))

# Imputation removed column names; put them back
imputed_x_train_plus.columns = x_train_plus.columns
imputed_x_valid_plus.columns = x_valid_plus.columns

print('*************************')
print('imputated w/ col names')
print('*************************')
print(imputed_x_train_plus.head(5))

print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_x_train_plus, imputed_x_valid_plus, y_train, y_valid))








# Shape of training data (num_rows, num_columns)
print(x_train.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (x_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])



# https://www.kaggle.com/alexisbcook/categorical-variables