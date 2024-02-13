'''
Assignment  : 3
Date        : 5 Jan 2024
Name        : Chonakan Chumtap 
Student ID  : 6410301022
'''

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

np.random.seed(1)

mean_A = np.arange(1, 101) * 0
mean_A[0] = 3
covariance_A = np.eye(100, 100, dtype = int) * 0.75
covariance_A[0] = 0.10

num_sample_classA = 250

mean_B = np.arange(1, 101) * 0
mean_B[0] = 6
covariance_B = np.eye(100, 100, dtype = int) * 0.75

num_sample_classB = 250


classA_data = np.random.multivariate_normal(mean_A, covariance_A, num_sample_classA)
classB_data = np.random.multivariate_normal(mean_B, covariance_B, num_sample_classB)

test_data = np.vstack((classA_data, classB_data))
label = np.hstack((np.zeros(num_sample_classA), np.ones(num_sample_classB))) # makes class A's label to be 1 and class B's label to be 0


# Calculate Class Probability
# P(A): Prob of data A => Y
prior_classA = len(classA_data) / (len(classA_data) + len(classB_data))
# P(B): Prob of data B => Y
prior_classB = len(classB_data) / (len(classA_data) + len(classB_data))



