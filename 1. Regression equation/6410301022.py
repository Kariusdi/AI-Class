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

        sum_x, sum_y = 0, 0

        for i,j in zip(self.x, self.y):
            sum_x += i
            sum_y += j
        
        x_bar = sum_x / len(self.x)
        y_bar = sum_y / len(self.y)

        return x_bar, y_bar
    
    def find_sum_x_y(self):

        x_bar, y_bar = self.find_xy_bar()
        Sxx, Syy, Sxy = 0.0, 0.0, 0.0

        for x,y in zip(self.x, self.y):
            Sxx += round((x - x_bar) ** 2, 1)
            Syy += round((y - y_bar) ** 2, 1)
            Sxy += round((x - x_bar) * (y - y_bar), 1)
        
        return Sxx, Syy, Sxy
 
    def find_ab(self):
        
        x_bar, y_bar = self.find_xy_bar()
        Sxx, Syy, Sxy = self.find_sum_x_y()

        a = round(Sxy / Sxx, 1)
        b = round(y_bar - (x_bar * a), 1)

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
