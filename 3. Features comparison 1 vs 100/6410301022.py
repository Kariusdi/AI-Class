'''
Assignment  : 3
Date        : 5 Jan 2024
Name        : Chonakan Chumtap 
Student ID  : 6410301022
'''

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# Step 0: Generate Data
np.random.seed(1)
# mean_A = [3, 0]
# covariance_A = np.array([[0.10, 0.00], [0.00, 0.75]])

# B-100 features
mean_A = np.arange(1, 101) * 0
mean_A[0] = 3
covariance_A = np.eye(100, 100, dtype = int) * 0.75
covariance_A[0][0] = 0.10
num_sample_classA = 250

# mean_B = [6, 0]
# covariance_B = np.array([[0.75, 0.00], [0.00, 0.75]])

# A-100 features
mean_B = np.arange(1, 101) * 0
mean_B[0] = 6
covariance_B = np.eye(100, 100, dtype = int) * 0.75
num_sample_classB = 250

# Step 1: Generate data
classA_data = np.random.multivariate_normal(mean_A, covariance_A, num_sample_classA)
classB_data = np.random.multivariate_normal(mean_B, covariance_B, num_sample_classB)

# Step 2: Calculate Class-Conditional Probabilities
pdf_A = multivariate_normal(mean=mean_A, cov=covariance_A)
pdf_B = multivariate_normal(mean=mean_B, cov=covariance_B)


#----------------- Step 3: Bayes Decision Theory -----------------#

# Calculate Class Probability
# P(A): Prob of data A => Y
prob_A = len(classA_data) / (len(classA_data) + len(classB_data))
# P(B): Prob of data B => Y
prob_B = len(classB_data) / (len(classA_data) + len(classB_data))

def bayes_classifier(sample):
    # A Sample (X)
    features = sample
    
    # Split 2 features from using pdf to define condition prob
    pxy_A = pdf_A.pdf(features)
    pxy_B = pdf_B.pdf(features)

    # Bayes formular: P(Y|X) = P(X|Y)P(Y) / P(X)
    # which P(X) is from sum of P(X|Y)P(Y)
    px = (pxy_A * prob_A) + (pxy_B * prob_B)

    # P(Y|X) = P(X|Y)P(Y) / P(X)
    pyxA = (pxy_A * prob_A) / px
    pyxB = (pxy_B * prob_B) / px

    return pyxA, pyxB

#-----------------------------------------------------------------#

# Step 4: Make Predictions
A_predictions = []
A_error = 0
for sampleA in np.concatenate([classA_data]):
    # Posterior of A and B
    IsA, IsB = bayes_classifier(sampleA)

    if IsA > IsB:
        predicted_class = 'A'
    else:
        predicted_class = 'B'
        A_error += 1
    A_predictions.append(predicted_class)
# print(A_predictions)

B_predictions = []
B_error = 0
for sampleB in np.concatenate([classB_data]):
    # Posterior of A and B
    IsA, IsB = bayes_classifier(sampleB)

    if IsA > IsB:
        predicted_class = 'A'
        B_error += 1
    else:
        predicted_class = 'B'
    B_predictions.append(predicted_class)
# print(B_predictions)

print("Error from Data A : {}".format(A_error))
print("Error from Data B : {}\n".format(B_error))
print("Total Error : {}".format(A_error + B_error))

# Plotting data
# plt.scatter(classA_data[:, 0], classA_data[:, 1], marker='.', s=50, alpha=0.5, color='red', label = 'a')
# plt.scatter(classB_data[:, 0], classB_data[:, 1], marker='.', s=50, alpha=0.5, color='blue', label = 'b')

# plt.axis('equal')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.xlim(-10, 10)
# plt.ylim(-10, 10)
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# plt.legend()
# plt.grid()
# plt.show()