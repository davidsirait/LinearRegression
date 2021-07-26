import numpy as np


class CostFunction:
    """

    Class for different metrics on calculating errors and cost functions.

    Attributes
    ----------
    y_target : array of the real value of dependent variable

    y_predict : array of predicted value of y


    """
    def __init__(self, y_target, y_predict):
        self.y_target = y_target
        self.y_predict = y_predict

    def mean_squared_error(self):
        """
        calculate the mean squared error :
            mse = sum (y_target(i) - y_predict(i)) ^2 / n_target
            where y_target(i) and y_predict(i) is the i-th element of the array

        :return:  mean squared error
        """
        return np.average((self.y_target - self.y_predict) ** 2, axis=0)

    def mean_absolute_error(self):
        return np.average(abs(self.y_target - self.y_predict))

    def root_mean_squared_error(self):
        """
        the square root of the mean squared error :
            rmse = sqrt(mse)

        :return:  root mean squared error
        """
        return self.mean_squared_error() **0.5

    def r_squared(self):
        """
        measure the r^2 or coefficient of determination, which describes proportion
        of variation in the dependent variable that is predictable given the independent
        variable(s) , and can be used to measure how well a regression model approximate
        the real data points.

        r^2 commonly comes in a scale 0 to 1, with value of 1 a regression which perfectly
        fit the data, and vice versa.

        r squared is given bv :

                        r^2 = 1 - SSE / SStot

        where SSE is the residual sum of squares and SStot is the total sum of squares

        :return : r squared value
        """
        ss_res = np.sum((self.y_target - self.y_predict) ** 2)
        ss_tot = np.sum((self.y_target - self.y_target.mean()) ** 2)

        return 1 - ss_res / ss_tot


