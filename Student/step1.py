import pandas as pd

# set display options to output to the console in pycharm
pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 50)

# Path of the file to read
iowa_file_path = 'train.csv'

data = pd.read_csv(iowa_file_path)

print(data.describe())

# is there any missing data?
# let's see
print('Count before dropping any rows with null values:')
print(len(data))
print('Count after dropping any rows with null values')
data = data.dropna(axis=1)
print(len(data))


# what is the average lot size?
# As of today, how old is the newest home?

# The newest house in your data isn't that new. A few potential explanations for this:
#
# They haven't built new houses where this data was collected.
# The data was collected a long time ago. Houses built after the data publication wouldn't show up.
# If the reason is explanation #1 above, does that affect your trust in the model you build with this data? What about if it is reason #2?
#
# How could you dig into the data to see which explanation is more plausible?