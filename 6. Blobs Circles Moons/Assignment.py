import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras import layers

def generateData(data_pattern):
    df_data = pd.DataFrame()
    if data_pattern=='blobs':
        X_train, y_train = make_blobs(n_samples=100,
                              n_features=2,
                              centers=2,
                              cluster_std=0.2,
                              center_box=(0,5))
    elif data_pattern=='circles':
        X_train, y_train = make_circles(n_samples=100,
                              noise=0.1,
                              factor=0.2)
    elif data_pattern=='moons':
        X_train, y_train = make_moons(n_samples=100,
                              noise=.05)
    df_data['x'] = X_train[:,0]
    df_data['y'] = X_train[:,1]
    df_data['cluster'] = y_train
    return df_data

def preparing_data(dat):
    ss = StandardScaler()
    data_ss = ss.fit_transform(dat)
    return data_ss

def buildModel(data_pattern):

    if data_pattern == 'blobs':
        model = Sequential([
            layers.Dense(32, activation="relu", input_shape=(2,)),
            layers.Dense(1, activation="sigmoid")
        ])
    elif data_pattern == 'circles':
        model = Sequential([
            layers.Dense(32, activation="relu", input_shape=(2,)),
            layers.Dense(1, activation="sigmoid")
        ])
    elif data_pattern == 'moons':
        model = Sequential([
            layers.Dense(32, activation="relu", input_shape=(2,)),
            layers.Dense(1, activation="sigmoid")
        ])

    return model

def trainingModel(dataX, dataY, model):
    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

    X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=0.25, random_state=1)

    start = time.time()
    history = model.fit(X_train, Y_train, epochs=10, verbose=1, batch_size=25, validation_split=0.25)
    end = time.time()
    print("Training Time: {:.3f} secs".format(end-start))

    return model, X_test, Y_test

def testModel(x_test, y_test):

    score = model.evaluate(x_test, y_test, verbose=0)
    return score

    # y_pred_prob = model.predict(x_test)
    # y_pred = np.where(y_pred_prob>0.5, 1, 0)

    # print(y_pred[:5])

def plot_decision_boundary(model, X, Y):
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

if __name__ == "__main__":
    
# ------------------ Blobs Data ------------------ #
    
    data_blobs = generateData("blobs")
    blobs_x = data_blobs.drop('cluster', axis=1)
    blobs_y = data_blobs['cluster']

    # 1). Prepare Data
    X = preparing_data(blobs_x)
    Y = blobs_y.values.reshape(-1, 1) # transpose to resize

    # 2). Build Model
    model_blobs = buildModel()

    # 3). Complie and Train Model
    model, x_test, y_test = trainingModel(X, Y, model_blobs)

    # 4). Evaluation and Testing Model
    score = testModel(x_test, y_test)
    print("Loss: ", score[0])
    print("Accuracy: ", score[1])

    # 5). Save Model and Scaler
    # model.save('blobsData_model.keras')
    plot_decision_boundary(model, X, Y)

# ----------------------------------------------- #