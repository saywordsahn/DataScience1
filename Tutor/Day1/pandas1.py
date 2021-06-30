import pandas as pd

pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 50)

file_path = 'melb_data.csv'

data = pd.read_csv(file_path)
data = data.dropna(axis=0)
# print(data.describe())
# print(data.columns)
# print(data.SellerG)

y = data.Price

features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

x = data[features]

print(x.describe())
print(x.head(10))