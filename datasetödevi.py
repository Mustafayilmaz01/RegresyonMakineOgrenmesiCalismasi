import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet,LassoCV,RidgeCV,ElasticNetCV
from sklearn.metrics import mean_squared_error,r2_score



df=pd.read_csv("summ.csv")
print(df.head())
print(df.describe())
print(df.info())
print(df.columns)
df.rename(columns={
    'Temperature (C)': 'Temperature',
    'Apparent Temperature (C)': 'Apparent Temperature',
    'Wind Speed (km/h)': 'Wind Speed',
    'Wind Bearing (degrees)': 'Wind Bearing',
    'Visibility (km)': 'Visibility',
    'Pressure (millibars)': 'Pressure'
}, inplace=True)
df.drop(columns=["Formatted Date","Summary","Daily Summary"], inplace=True)
print(df.head())
print(df.info())
print(df.columns)

print(df["Precip Type"].value_counts())

from sklearn.preprocessing import StandardScaler,LabelEncoder

le=LabelEncoder()
le.fit(df["Precip Type"])
df["Precip Type"]=le.transform(df["Precip Type"])
df.dropna()
#print(df["Precip Type"].value_counts())
#print(df.info())

"""sns.heatmap(df.corr(),annot=True)
plt.show()"""


X=df.drop(columns=["Temperature","Apparent Temperature","Loud Cover"],axis=1)
y=df["Temperature"]
sns.heatmap(X.corr(),annot=True)
plt.show()
print("---")
print(df["Loud Cover"].unique())
print(X.head())

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=123)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

linear=LinearRegression()
linear.fit(X_train,y_train)
y_pred=linear.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Linear Regression Mean Squared Error",mse)
print("Linear Regression R2=Score",r2)
lasso=Lasso()
lasso.fit(X_train,y_train)
y_pred=lasso.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("-----------------------------------")

print("Lasso Linear Regression Mean Squared Error",mse)
print("Lasso Linear Regression R2=Score",r2)
ridge=Ridge()
ridge.fit(X_train,y_train)
y_pred=ridge.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("-----------------------------------")

print("Ridge Regression Mean Squared Error",mse)
print("Ridge Regression R2=Score",r2)
yeniveriseti=df.copy()
print("-----------------------------------")
ridgecv=RidgeCV(cv=5)
ridgecv.fit(X_train,y_train)
y_pred=ridgecv.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("RidgeCV Regression Mean Squared Error",mse)
print("RidgeCV Regression R2=Score",r2)
print("-----------------------------------")


X=yeniveriseti.drop(columns=["Precip Type","Temperature"])
y=yeniveriseti["Temperature"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=123)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

linear=LinearRegression()
linear.fit(X_train,y_train)
y_pred=linear.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Yeni Linear Regression Mean Squared Error",mse)
print("Yeni Linear Regression R2=Score",r2)
print("-----------------------------------")

lasso=Lasso()
lasso.fit(X_train,y_train)
y_pred=lasso.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Yeni Lasso Linear Regression Mean Squared Error",mse)
print("Yeni Lasso Linear Regression R2=Score",r2)
print("-----------------------------------")

ridge=Ridge()
ridge.fit(X_train,y_train)
y_pred=ridge.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Yeni Ridge Regression Mean Squared Error",mse)
print("Yeni Ridge Regression R2=Score",r2)
print("-----------------------------------")
ridgecv=RidgeCV(cv=5)
ridgecv.fit(X_train,y_train)
y_pred=ridgecv.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Yeni RidgeCV Regression Mean Squared Error",mse)
print("Yeni RidgeCV Regression R2=Score",r2)

