#importing the necessary libraries
import pandas as pd
import sys
sys.path.append('C:/Users/Wlink/anaconda3/Lib/site-packages')
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error

model = joblib.load('../model/linear_regression_model.pkl') #loading the trained linear regression model

x_test=pd.read_csv('./data/X_test.csv')
y_test=pd.read_csv('./data/Y_test.csv')

y_pred=model.predict(x_test) #using the linear regression model to predict the target variable

x_test.head()

y_test.head()

#calculation of mean squared error, root mean squared error, r-squared, and mean absolute error
mse=mean_squared_error(y_test,y_pred) 
r2=r2_score(y_test,y_pred)
rmse = root_mean_squared_error(y_test,y_pred)
mae= mean_absolute_error(y_test,y_pred)

# %%
print(f"Mean squared error : {mse}\nR-squared error : {r2}\nRoot mean squared error : {rmse}\nMean absolute error : {mae}")

# %% [markdown]
# """
# Model Evaluation Summary
# Mean Squared Error (MSE): 12.92
# 
# Measures the average squared difference between actual and predicted values. Lower is better.
# 
# Root Mean Squared Error (RMSE): 3.59
# 
# Square root of MSE; more interpretable as it’s in the same unit as the target variable (housing prices).
# 
# Mean Absolute Error (MAE): 2.55
# 
# Average of absolute errors; shows on average how much predictions deviate from actual values.
# 
# R² Score (R-squared): 0.764
# 
# The model explains 76.4% of the variance in the target variable, which is quite solid for this kind of data. A good R² above 0.75 indicates it's capturing most of the underlying patterns in the data. 
# """

# %%



