# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class Polynomial_reg:
    def __init__(self,data):
        self.data=data
        # Importing the dataset
        dataset = pd.read_csv(data)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, self.y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        
        # Training the Polynomial Regression model on the Training set
        poly_reg = PolynomialFeatures(degree = 4)
        X_poly = poly_reg.fit_transform(X_train)
        regressor = LinearRegression()
        regressor.fit(X_poly, y_train)
        
        # Predicting the Test set results
        self.y_pred = regressor.predict(poly_reg.transform(X_test))
        np.set_printoptions(precision=2)
        print(np.concatenate((self.y_pred.reshape(len(self.y_pred),1), self.y_test.reshape(len(self.y_test),1)),1))
    
    def report_r2(self):
    # Evaluating the Model Performance
        print(r2_score(self.y_test, self.y_pred))