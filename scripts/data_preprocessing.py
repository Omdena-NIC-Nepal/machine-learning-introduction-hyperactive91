import pandas as pd
import sys
sys.path.append('C:/Users/Wlink/anaconda3/Lib/site-packages')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv('../data/BostonHousing.csv')
df.head(3)
df.info()

# # Handling Outliers :
# Using interquartile range to find the number of outliers in each column.
def count_outliers_IQR(data):
    outliers={}
    for col in data.select_dtypes(include=['number']).columns:
        q1=data[col].quantile(0.25)
        q3=data[col].quantile(0.75)
        IQR=q3-q1 #formula to calculate the difference between 3rd and 1st quartile
        low_bound=q1-1.5*IQR
        upp_bound=q3+1.5*IQR
        outlier_count=data[(data[col]<low_bound)|(data[col]>upp_bound)][col].count()
        if outlier_count>0:
            outliers[col]=int(outlier_count)
    return outliers

outliers_found = count_outliers_IQR(df)
print("Outlier counts per column:", (outliers_found))
        

# # Using standard to scale the data and isolation forest to remove the outliers

from sklearn.ensemble import IsolationForest
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Applying IsolationForest

iso = IsolationForest(contamination=0.05, random_state=42) #using 7% contamination 
out_liers = iso.fit_predict(df_scaled)

# Tag and remove outliers
df['outlier'] = out_liers
df_cleaned = df[df['outlier'] != -1].drop(columns='outlier')

#to visualize boxplot before and after outlier is removed
fig,ax=plt.subplots(1,2,figsize=(14,8))
ax[0].boxplot(df)
ax[0].set_title('Features and target variable before outliers is removed')

ax[1].boxplot(df_cleaned)
ax[1].set_title('Features and target variable after outliers is removed')

plt.show()


# Print the shape before and after
print(f"Original shape: {df.shape}")
print(f"Cleaned shape: {df_cleaned.shape}")

# Splitting the dataset
X = df_cleaned.drop(columns=['medv'])
y = df_cleaned['medv']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Saving the preprocessed data
import os
os.makedirs("./data", exist_ok=True)

# Saving the datasets 
X_train.to_csv("./data/X_train.csv", index=False)
X_test.to_csv("./data/X_test.csv", index=False)
y_train.to_csv("./data/y_train.csv", index=False)
y_test.to_csv("./data/y_test.csv", index=False)
print("done")





