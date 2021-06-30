# work through subsets problem
# try problem on leetcode?
# iterative or recursive solution okay!

# simple recursion example, sum first n elements of the fibonacci sequence
def fib(n):
   if n <= 1:
       return n
   else:
       return(fib(n-1) + fib(n-2))

print(fib(7))

# DO: implement our subsets in our algorithm to futher lower our MAE for the decision tree model
# DO: create our final model



# wrap up decision tree example, move to random forest
# do change model to random forest
from sklearn.ensemble import RandomForestRegressor

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)

# fit your model
# rf_model.fit(train_X, train_y)

# Calculate the mean absolute error of your Random Forest model on the validation data
# rf_val_mae = mean_absolute_error(val_y, rf_model.predict(val_X))

# lets make our best model

