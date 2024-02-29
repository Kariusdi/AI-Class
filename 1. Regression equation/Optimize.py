'''
Assignment  : 1
Date        : 2 Dec 2023
Name        : Chonakan Chumtap 
Student ID  : 6410301022
'''

import matplotlib.pyplot as plt
import numpy as np

class Regression():

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def find_xy_bar(self):

        x_bar = np.average(self.x)
        y_bar = np.average(self.y)

        return x_bar, y_bar
    
    def find_sum_x_y(self):

        x_bar, y_bar = self.find_xy_bar()

        Sxx = (self.x - x_bar) ** 2
        Syy = (self.y - y_bar) ** 2
        Sxy = (self.x - x_bar) * (self.y - y_bar)
        
        return Sxx, Syy, Sxy
 
    def find_ab(self):
        
        x_bar, y_bar = self.find_xy_bar()
        Sxx, Syy, Sxy = self.find_sum_x_y()

        a = np.sum(Sxy) / np.sum(Sxx)
        b = np.sum(y_bar) - (np.sum(x_bar) * a)

        return a, b
    
    def equation(self):

        a, b = self.find_ab()
        y = a * self.x + b

        return y

if __name__ == '__main__':

    # x = np.array([i for i in range (4,20)])
    # y = np.array([100.1, 107.2, 114.1, 121.7, 126.8, 130.9, 137.5, 143.2, 149.4, 151.1, 154.0, 154.6, 155.0, 155.1, 155.3, 155.7])
    
    x = np.array([29, 28, 34, 31, 25])         # x
    y = np.array([77, 62, 93, 84, 59])         # y     

    regression = Regression(x, y)

    # Regression equation
    y_eq = regression.equation()

    plt.xlabel("High temp in °C")
    plt.ylabel("Iced tea orders")
    plt.scatter(x, y, color='darkorange')       # Scatter plot
    plt.plot(x, y_eq, label='regression line')  # Line plot
    plt.legend()
    plt.show()
