# This is a machine learning code based on scikit-learn library, 
# designed to predict house prices based on Ames Housing dataset.
#------------------------------------------------------------------------------
# Machine Learning approach:---------------------------------------------------
# We use two models (Descision Tree Regressor and Random Forest) to train data
# and predict house prices. The models are validated by mean absolute error, 
# and the results of the model with the lowest error are expored in a output file

# Origin of training data sets:------------------------------------------------
#   The Ames Housing dataset was compiled by Dean De Cock for use in data science education.
# It's an incredible alternative for data scientists looking for a modernized and expanded
# version of the often cited Boston Housing dataset. 
#   For more information please visit:
# http://jse.amstat.org/v19n3/decock.pdf
# https://www.kaggle.com/c/home-data-for-ml-course/overview


# Input and Output:------------------------------------------------------------
# All input and output files have to be in the root directory
# Input1: training data sets are imported from file called 'train.csv'
# Input2: training data sets are imported from file called 'test.csv'
# Output: file 'predictedprices.csv' with the predicted house prices:
#                   1st coloumn containt ID; 2nd coloumn has the predicted price.

#----Begin---------------------------------------------------------------------
import pandas as pd
#from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Access the test data sets:
test_data_path = 'test.csv'
# read test data file using pandas:
test_data = pd.read_csv(test_data_path)
# replace all values NaN to -111.111
test_data = test_data.fillna(-111.111) 

# Access the train data sets:
train_file_path = 'train.csv'
# read test data file using pandas:
home_data = pd.read_csv(train_file_path)
# replace all values NaN to -111.111
home_data = home_data.fillna(-111.111) #replaces NaN to -111.111


# Read the titles of feature coloumns from the train data set 'home_data':
features_total=home_data.columns.values.tolist()
# Read the titles of feature coloumns from the test data set 'test_data':
features_test_total=test_data.columns.values.tolist();
# Pick only features from the test data that are quantified, i.e. that are expressed in numbers:
features_Num=[]
xtemp = []
for iitemp in range(0,len(features_test_total)) : 
    if is_numeric_dtype(test_data[features_test_total[iitemp]]) : 
        features_Num.append(features_test_total[iitemp])
        xtemp.append(iitemp)
features = features_Num


# Create target object and call it y
y = home_data.SalePrice

# Create train obeject and call it X
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Try two models on the training data, and pick up one that gives the smallest 
# error on validation data

# Model 1: Here Descision Tree Regressor is used to train and predict the house prices
# -----------------------------------------------------------------------------
# Specify Model:
dtr_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
# Fit Model:
dtr_model.fit(train_X, train_y)
# Predict house prices:
dtr_val_predictions = dtr_model.predict(val_X)
# Calculate the prediction error:
dtr_val_mae = mean_absolute_error(dtr_val_predictions, val_y)
# Show the error on the screen:
print("Validation MAE for Decision Tree Regressor: {:,.0f}".format(dtr_val_mae))

# Model 2: Here Random Forest model is used to train and predict the house prices
# -----------------------------------------------------------------------------
# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
# Fit Model:
rf_model.fit(train_X, train_y)
# Predict house prices:
rf_val_predictions = rf_model.predict(val_X)
# Calculate the prediction error:
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
# Show the error on the screen:
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))


# Pick up the best model (i.e. that gives smaller error) to train on full data
# and predict the house prices:
if dtr_val_mae > rf_val_mae:
  model_on_full_data = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
  print('Decision Tree Regressor shows better accuracy on the input data set and will be used to predict the house prices')
else :
  model_on_full_data = RandomForestRegressor(random_state=1)
  print('Random Forest Model shows better accuracy on the input data and will be used to predict the house prices')


# fit rf_model_on_full_data on all data from the training data:
model_on_full_data.fit(X, y)

## create test_X which comes from test_data but includes only the columns that are used for prediction.
## The list of columns is stored in a variable called features
test_X = test_data[features]

# predict the house prices based on the trained model
test_preds = model_on_full_data.predict(test_X)

# Export the predicted house prices into file with extension .csv
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('predictedprices.csv', index=False)

#----End----------------------------------------------------------------------




