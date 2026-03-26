# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the car dataset, preprocess the data by removing unwanted columns and converting categorical values into numerical form.

2.Separate features (X) and target variable (price), then normalize the data using StandardScaler.

3.Split the dataset into training and testing sets and train the SGD Regressor model using the training data.

4.Predict prices using the test data and evaluate model performance using MSE and R² score, then visualize the results.##

## Program:
```
/*
Program to implement SGD Regressor for linear regression.
Developed by: ROSHAN V
RegisterNumber: 25004228 // 212225240124
*/

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("caras.csv")
print(data.head())
print(data.info())


data=data.drop(['car_ID','CarName'],axis=1)
data=pd.get_dummies(data,drop_first=True)

X=data.drop('price',axis=1)
y=data['price']

scaler = StandardScaler()
X=scaler.fit_transform(X)
y =scaler.fit_transform(np.array(y).reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)

sgd_model.fit(X_train,y_train)

y_pred=sgd_model.predict(X_test)

mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print("Name: ROSHAN V ")
print("Reg No: 212225240124 ")
print("Mean Squared Error:",mse)
print("R-squared Score:",r2)

print("\nModel Coefficients:")
print("Coefficients:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)

plt.scatter(y_test,y_pred)
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color='red')
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.xlabel("Actual Price ")
plt.ylabel("Predicted Price ")
plt.grid(True)
plt.show()
```

## Output:
<img width="1127" height="751" alt="image" src="https://github.com/user-attachments/assets/81d89d97-6f35-436c-a2a1-479891245fab" />
<img width="1019" height="454" alt="image" src="https://github.com/user-attachments/assets/7e470f2e-eece-40f3-bb02-44cef91208e7" />




## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
