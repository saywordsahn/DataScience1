import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 50)

data = pd.read_csv('housing.csv')

print(data.head())
print(data.info())
print(data.describe())

# generate histograms for each variable
data.hist(bins=50, figsize=(20,15))
plt.show()

# split into training and test data sets
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
