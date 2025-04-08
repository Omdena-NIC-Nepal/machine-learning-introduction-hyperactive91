import pandas as pd
import sys
sys.path.append('C:/Users/Wlink/anaconda3/Lib/site-packages')
from sklearn.linear_model import LinearRegression
import joblib


# reading the preprocessed x_train and y_train csv file
x_train=pd.read_csv('./data/X_train.csv')
y_train=pd.read_csv('./data/Y_train.csv')


# training the model

model=LinearRegression() # choosing model for training 
model.fit(x_train,y_train) # training the model


import os
os.makedirs('../model',exist_ok=True) #making directory named model
joblib.dump(model,'../model/linear_regression_model.pkl') #dumping the model in that directory





