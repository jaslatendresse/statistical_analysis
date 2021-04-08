import pandas as pd
import scipy 
import numpy as np
import math
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

#Linear regression
def regression():
    """
    Example (linear regression): predict how much snow you think will fall this year. 
    Data type: X, Y 
    """
    lin_data = pd.read_csv('data/lin_regression.csv')
    X = lin_data.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = lin_data.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    model = linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.show()
    r_squared = model.score(X,Y)
    print(r_squared)

