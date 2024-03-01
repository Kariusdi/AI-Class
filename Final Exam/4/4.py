from matplotlib.colors import ListedColormap
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras import layers

class Classifying:

    def __init__(self):
        pass
    
    def preparing_data(self, dat):
        ss = StandardScaler()
        feature_ss = ss.fit_transform(dat)
        return feature_ss
    
    def buildModel(self, shape):
        model = Sequential([
            layers.Dense(8, activation="tanh", input_shape=(shape,)),
            layers.Dense(4, activation="relu"),
            layers.Dense(4, activation="relu"),
            layers.Dense(1, activation="sigmoid")
        ])
        return model

    def trainingModel(self, model, dataX, dataY):
        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        
        X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=0.5, random_state=1)

        start = time.time()
        history = model.fit(X_train, Y_train, epochs=70, verbose=1, batch_size=25, validation_split=0.25)
        end = time.time()
        print("Training Time: {:.3f} secs".format(end-start))

        return model, X_test, Y_test
    
    def testModel(self, model, x_test, y_test):
        score = model.evaluate(x_test, y_test, verbose=0)
        return score
    
    def saveModel(self, model, path):
        model.save(path)
        print("Model saved successfully")
    
    def plot_decision_boundary(self, model, X, Y):
        h = .02  # Step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(y_min, y_max, h), np.arange(x_min, x_max, h))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        cmap_background = ListedColormap(['#FFAAAA', '#AAAAFF'])
        plt.contourf(xx, yy, Z, cmap=cmap_background, alpha=0.3)
        
        cmap_points = ListedColormap(['#FF0000', '#0000FF'])
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_points, edgecolors='k', marker='o')

        plt.title('Decision Boundary')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

if __name__ == '__main__':

    # 0). Genarate Data Set
    np.random.seed(0)

    mean1_1 = [2, 2]
    mean1_2 = [8, 8]
    mean2_1 = [5, 5]
    mean2_2 = [11, 11]

    cov1 = np.array([[1, 0.5], [0.5, 1]])
    cov2 = np.array([[3, -2.5], [-2.5, 3]])

    sample1_1 = np.random.multivariate_normal(mean1_1, cov1, size=100)
    sample1_2 = np.random.multivariate_normal(mean1_2, cov1, size=100)
    sample2_1 = np.random.multivariate_normal(mean2_1, cov2, size=100)
    sample2_2 = np.random.multivariate_normal(mean2_2, cov2, size=100)

    sample1 = np.concatenate((sample1_1, sample1_2), axis=0)
    sample2 = np.concatenate((sample2_1, sample2_2), axis=0)

    x = np.concatenate((sample1, sample2), axis=0)

    classes = Classifying()
    
    # 1). Prepare Data
    X = classes.preparing_data(x)
    y = np.concatenate((np.zeros(200), np.ones(200)))

     # 2). Build Model
    model_2classes = classes.buildModel(X.shape[1])

    # 3). Complie and Train Model
    model, x_test, y_test = classes.trainingModel(model_2classes, X, y)

    # 4). Evaluation and Testing Model
    score = classes.testModel(model, x_test, y_test)
    print("Loss: ", score[0])
    print("Accuracy: ", score[1])

    # 5). Plotting decision boundary
    classes.plot_decision_boundary(model, X, y)