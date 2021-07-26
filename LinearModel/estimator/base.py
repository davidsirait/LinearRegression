import numpy as np
from LinearModel.helpers.validation import *

class LinearRegr:
    """
    linear regression using ordinary least square method

    Fitting a linear model into the target dataset using the general formula :

            y_pred = w1 + w2*x1 + w3*x2 + ... + wP*xp-1

    where y_pred is the model predition of target/dependent variable and
    p-vector w = (w1,w2,w2,...,wp) is the coefficient. Length of vector w corresponds
    to the number of features given by the input dataset , and w1 is equal to the
    intercept.

    Attributes
    ----------
    tetha : array of linear coefficient w with length n_features.
            the very first elements of this array is the intercept

    intercept : float, the independent term which move the linear
                regresion line up or down along the y-axis. An intercept
                value of 0 means the regression line is passing through
                the origin

    coef_ : array of the coefficients, minus the intercept

    """
    def __init__(self):
        pass

    def fit(self, x, y):
        """
        fit a linear regression model

        :param x: numpy array or list of independent variables
        :param y: numpy array or list of target/ dependent variables
        :return: self : an instance of self
        """
        # check for appropriate input data type
        x_, y_ = check_is_array(x, y)

        # create a vandermonde matrix
        vandermonde = vandermonde_(x_)
        self.tetha = np.linalg.inv(vandermonde.T @ vandermonde) @ vandermonde.T @ y_
        self.intercept = self.tetha[0]
        self.coef_ = self.tetha[1:]

        return self

    def predict(self, x):
        """
        Calculates prediction of dependent variables

        :param x: numpy array or list of independent variables
        :return: prediction : array of y_predi values based on the input array
                              and fitted models coefficients
        """
        check_fit(self)
        prediction = vandermonde_(x) @ self.tetha

        return prediction

    def __repr__(self):
        return "An Ordinary Least Square Regression Model"


def vandermonde_(x):
    """
    Create a matrix consisting of ones in the first column with vector(s) of
    independent variable input on the subsequent columns , with number of repeating
    column of x vector depends on the number of features.

    :param x: numpy array or list of independent variables
    :return: matrix of shape (n_data, n_features + 1)
    """
    ones = np.ones((x.shape[0], 1))

    return np.hstack((ones, x))
