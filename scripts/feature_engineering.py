import sys
sys.path.append("C:/Users/Wlink/anaconda3/Lib/site-packages")
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns


###Load the Preprocessed Dataset
def read_file(filename):
    filepath = './data/'+str(filename)
    return pd.read_csv(filepath)

X_test = read_file('X_test.csv')
y_test = read_file('y_test.csv')
X_train = read_file('X_train.csv')
y_train = read_file('y_train.csv')

print(len(X_test), len(X_train), len(y_test), len(y_train))

print("X_test data: \n", X_test)
print("y_test data: \n", y_test)


# Create new interaction features
X_train['LSTAT_RM'] = X_train['lstat'] * X_train['rm']
X_test['LSTAT_RM'] = X_test['lstat'] * X_test['rm']

# Add polynomial feature (e.g., squared value)
X_train['RM_squared'] = X_train['rm'] ** 2
X_test['RM_squared'] = X_test['rm'] ** 2


# Train the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict
y_pred = lr_model.predict(X_test)


#Evaluate Model
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))


# Save the updated dataset with new features
joblib.dump({
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test
}, "../data/engineered_data.pkl")

# Save the model
joblib.dump(lr_model, "../data/linear_regression_with_features.pkl")


poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train with poly features
poly_lr = LinearRegression()
poly_lr.fit(X_train_poly, y_train)

# Evaluate
y_poly_pred = poly_lr.predict(X_test_poly)
print("Poly MSE:", mean_squared_error(y_test, y_poly_pred))
print("Poly R²:", r2_score(y_test, y_poly_pred))


# Save the model
joblib.dump(y_poly_pred, "../data/y_polymial_predication.csv")





