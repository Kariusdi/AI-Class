import numpy as np 
import math

def sigmoid(x):
    return (math.e**x) / (1+math.e**x)

def gradient(w, b, x, y):
    n = len(x)

    del_w1 = (2/n) * np.matmul(np.matmul(x, w) + b - y, x.T[0])
    del_w2 = (2/n) * np.matmul(np.matmul(x, w) + b - y, x.T[1])
    del_b = (2/n) * np.matmul(np.matmul(x, w) + b - y, np.ones([n]))

    return del_w1, del_w2, del_b


x = np.array( [ [1,1,1,0,1,1,0],
                [1,0,1,1,0,0,1],
                [1,0,0,0,1,1,0],
                [1,1,0,1,0,0,1],
                [1,1,1,0,1,0,1],
                [1,0,0,1,0,1,0],
                [1,0,1,0,1,0,1],
                [1,1,0,1,0,0,0],
                [1,0,1,0,1,1,1],
                [1,1,0,0,1,1,0]], dtype=np.float64)

y = np.array([[0],[0],[1],[0],[1],[0],[1],[0],[0],[1]])

weights = np.ones(x.shape).transpose()
lr = 0.01 # learning rate

for _ in range(2000):
    for idx, x_i in enumerate(x):
        # Forward (Forward propergation)
        linear_output = np.matmul(x_i, weights)
        activation_func = sigmoid(linear_output)
        # ----------------------------------------------------
        # Update weight (Back propergation)
        del_w = (2/len(x)) * np.matmul((activation_func - y) * linear_output * (1 - linear_output), x)
        weights = weights - (lr * del_w.transpose())
        
        error = sum(abs(activation_func - y))

print(len(weights))
for w in range(len(weights)):
    print('feature {} = {}'.format(w+1, np.mean(weights[w])))