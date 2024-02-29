'''
Assignment  : 3
Date        : 3 Jan 2023
Name        : Chonakan Chumtap 
Student ID  : 6410301022
'''

import numpy as np
import matplotlib.pyplot as plt

# one feature
mean1 = [3, 6]
cov1 = np.array([[0.10, 0.00], [0.00, 0.75]])
# 99 features
mean2 = [0, 0]
cov2 = np.array([[0.75, 0.00], [0.00, 0.75]])

pts1 = np.random.multivariate_normal(mean1, cov1, size=100)
pts2 = np.random.multivariate_normal(mean2, cov2, size=100)

plt.scatter(pts1[:, 0], pts1[:, 1], marker='.', s=50, alpha=0.5, color='red', label = 'a')
plt.scatter(pts2[:, 0], pts2[:, 1], marker='.', s=50, alpha=0.5, color='blue', label = 'b')

plt.axis('equal')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(-10, 10)
plt.ylim(-10, 10)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.legend()
plt.grid()
plt.show()