'''
Assignment  : 4
Date        : 8 Jan 2024
Name        : Chonakan Chumtap 
Student ID  : 6410301022
'''

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

mean1 = [3, 6]
cov1 = np.array([[0.10, 0.00], [0.00, 0.75]])

mean2 = [0, 0]
cov2 = np.array([[0.75, 0.00], [0.00, 0.75]])

dataA = np.random.multivariate_normal(mean1, cov1, size=10)
dataB = np.random.multivariate_normal(mean2, cov2, size=10) 

weight0 = 1
weight1 = 1
bias = 0.5

step_size = 0.7

start_line = weight0*dataA[0] + weight1*dataB[0] - bias
print(start_line)


# plt.scatter(start_line[0], start_line[1], marker='.', s=50, alpha=0.5, color='green')

# xpoints = np.array([weight1, 10])
# ypoints = np.array([weight0, 10])

# plt.plot(xpoints, ypoints)


plt.scatter(dataA[:, 0], dataA[:, 1], marker='.', s=50, alpha=0.5, color='red', label = 'a')
plt.scatter(dataB[:, 0], dataB[:, 1], marker='.', s=50, alpha=0.5, color='blue', label = 'b')

plt.axis('equal')
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim(-8, 8)
plt.ylim(-8, 8)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.legend()
plt.grid()
plt.show()