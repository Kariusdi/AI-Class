from matplotlib.colors import ListedColormap
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
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
            layers.Dense(128, activation="relu", input_shape=(shape,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            # layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid")
        ])
        return model

    def trainingModel(self, model, dataX, dataY):
        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        
        X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=0.5, random_state=1)

        start = time.time()
        history = model.fit(X_train, Y_train, epochs=50, verbose=1, batch_size=10, validation_split=0.25)
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
    
    def plot_roc_curve(self, model, x_test, y_test):
        y_pred_prob = model.predict(x_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = roc_auc_score(y_test, y_pred_prob)

        plt.plot(fpr, tpr, label='ROC curve')
        plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='best')

        # Add ROC AUC value to the plot
        plt.text(0.6, 0.2, f'ROC AUC = {roc_auc:.2f}', fontsize=12)

        plt.show()

def generate_data(num_samples, start=0, num_turns=3, noise=0.5):
    t = np.linspace(0, 2 * np.pi * num_turns, num_samples)

    x = t * np.cos(t + start)
    y = t * np.sin(t + start)

    x += noise * np.random.randn(num_samples)
    y += noise * np.random.randn(num_samples)

    return np.column_stack((x,y))

if __name__ == '__main__':

    # 0). Genarate Data Set
    np.random.seed(0)
    num_samples = 1000
    sample1 = generate_data(num_samples)
    sample2 = generate_data(num_samples, np.pi)

    x = np.vstack((sample1, sample2))
    class1_y = np.zeros((num_samples, 1))  # Label for class 1 is 0
    class2_y = np.ones((num_samples, 1))  # Label for class 2 is 1

    classes = Classifying()
    
    # 1). Prepare Data
    X = classes.preparing_data(x)
    y = np.vstack((class1_y, class2_y))

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

    # 6). Plotting ROC Curve
    classes.plot_roc_curve(model, x_test, y_test)