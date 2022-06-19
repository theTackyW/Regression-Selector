# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 15:54:59 2022

@author: Jacky
"""

# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class reg:
    def __init__(self,data):
        self.data = data
        # Importing the dataset
        dataset = pd.read_csv(data)
        self.X = dataset.iloc[:, :-1].values
        self.y = dataset.iloc[:, -1].values
        self.split()
        self.train_predict()
    # Splitting the dataset into the Training set and Test set
    def split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 0)
    def report_r2(self):
    # Evaluating the Model Performance
        print(r2_score(self.y_test, self.y_pred))
        
class Polynomial_reg(reg):

    # Training the Polynomial Regression model on the Training set
    def train_predict(self):
        poly_reg = PolynomialFeatures(degree = 4)
        self.X_poly = poly_reg.fit_transform(self.X_train)
        regressor = LinearRegression()
        regressor.fit(self.X_poly, self.y_train)
        
        # Predicting the Test set results
        self.y_pred = regressor.predict(poly_reg.transform(self.X_test))
        np.set_printoptions(precision=2)
        #print(np.concatenate((self.y_pred.reshape(len(self.y_pred),1), self.y_test.reshape(len(self.y_test),1)),1))

class multi_linear_reg(reg):
    
        # Training the Multiple Linear Regression model on the Training set
    def train_predict(self):
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(self.X_train, self.y_train)
        
        # Predicting the Test set results
        self.y_pred = regressor.predict(self.X_test)
        np.set_printoptions(precision=2)
        #print(np.concatenate((self.y_pred.reshape(len(self.y_pred),1), self.y_test.reshape(len(self.y_test),1)),1))
class decision_tree_reg(reg):
    
    def train_predict(self):
        # Training the Decision Tree Regression model on the Training set
        from sklearn.tree import DecisionTreeRegressor
        regressor = DecisionTreeRegressor(random_state = 0)
        regressor.fit(self.X_train, self.y_train)
        
        # Predicting the Test set results
        self.y_pred = regressor.predict(self.X_test)
        np.set_printoptions(precision=2)
        #print(np.concatenate((self.y_pred.reshape(len(self.y_pred),1), self.y_test.reshape(len(self.y_test),1)),1))

class svr(reg):
    
    def split(self):
        self.y = self.y.reshape(len(self.y),1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 0)
    # Feature Scaling
    def train_predict(self):
        
        from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        self.X_train = sc_X.fit_transform(self.X_train)
        self.y_train = sc_y.fit_transform(self.y_train)
        
        # Training the SVR model on the Training set
        from sklearn.svm import SVR
        regressor = SVR(kernel = 'rbf')
        regressor.fit(self.X_train, self.y_train)
        
        # Predicting the Test set results
        self.y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(self.X_test)))
        np.set_printoptions(precision=2)
        #print(np.concatenate((self.y_pred.reshape(len(self.y_pred),1), self.y_test.reshape(len(self.y_test),1)),1))
        
class random_forest_reg(reg):
    def train_predict(self):
    # Training the Random Forest Regression model on the whole dataset
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
        regressor.fit(self.X_train, self.y_train)
        
        # Predicting the Test set results
        self.y_pred = regressor.predict(self.X_test)
        np.set_printoptions(precision=2)
        #print(np.concatenate((self.y_pred.reshape(len(self.y_pred),1), self.y_test.reshape(len(self.y_test),1)),1))