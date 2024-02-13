'''
Assignment  : Classifying 3 data patterns
Date        : 6 Feb 2024
Name        : Chonakan Chumtap 
Student ID  : 6410301022
'''

class Classifying:

    def __init__(self, data_pattern):
        self.data_pattern = data_pattern
        
    def generateData(self):
        df_data = pd.DataFrame()
        if self.data_pattern=='blobs':
            X_train, y_train = make_blobs(n_samples=100,
                                n_features=2,
                                centers=2,
                                cluster_std=0.2,
                                center_box=(0,5))
        elif self.data_pattern=='circles':
            X_train, y_train = make_circles(n_samples=100,
                                noise=0.1,
                                factor=0.2)
        elif self.data_pattern=='moons':
            X_train, y_train = make_moons(n_samples=100,
                                noise=.05)
        df_data['x'] = X_train[:,0]
        df_data['y'] = X_train[:,1]
        df_data['cluster'] = y_train
        return df_data
    
    def preparing_data(self, dat):
        ss = StandardScaler()
        feature_ss = ss.fit_transform(dat)
        return feature_ss
    
    def buildModel(self):

        if self.data_pattern == 'blobs':
            model = Sequential([
                layers.Dense(32, activation="relu", input_shape=(2,)),
                layers.Dense(1, activation="sigmoid")
            ])
        elif self.data_pattern == 'circles':
            model = Sequential([
                layers.Dense(64, activation="relu", input_shape=(2,)),
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(8, activation="relu"),
                layers.Dense(1, activation="sigmoid")
            ])
        elif self.data_pattern == 'moons':
            model = Sequential([
                layers.Dense(128, activation="relu", input_shape=(2,)),
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(1, activation="sigmoid")
            ])

        return model

    def trainingModel(self, model, dataX, dataY):
        
        if self.data_pattern == 'blobs':
            model.compile(loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])
            
            X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=0.25, random_state=1)

            start = time.time()
            history = model.fit(X_train, Y_train, epochs=20, verbose=1, batch_size=25, validation_split=0.25)
            end = time.time()
            print("Training Time: {:.3f} secs".format(end-start))
        elif self.data_pattern == 'circles':
            model.compile(loss='mean_squared_error',
                        optimizer='adam',
                        metrics=['accuracy'])
            
            X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=0.5, random_state=1)

            start = time.time()
            history = model.fit(X_train, Y_train, epochs=35, verbose=1, batch_size=10, validation_split=0.25)
            end = time.time()
            print("Training Time: {:.3f} secs".format(end-start))
        elif self.data_pattern == 'moons':
            model.compile(loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])
            
            X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=0.25, random_state=1)

            start = time.time()
            history = model.fit(X_train, Y_train, epochs=60, verbose=1, batch_size=10, validation_split=0.25)
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
    


if __name__ == "__main__":

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

# --------------------- Blobs --------------------- #
    
    blobs = Classifying("blobs")

    data_blobs = blobs.generateData()
    blobs_x = data_blobs.drop('cluster', axis=1)
    blobs_y = data_blobs['cluster']

    # 1). Prepare Data
    X = blobs.preparing_data(blobs_x)
    Y = blobs_y.values.reshape(-1, 1) # transpose to resize

    # 2). Build Model
    model_blobs = blobs.buildModel()

    # 3). Complie and Train Model
    model, x_test, y_test = blobs.trainingModel(model_blobs, X, Y)

    # 4). Evaluation and Testing Model
    score = blobs.testModel(model, x_test, y_test)
    print("Loss: ", score[0])
    print("Accuracy: ", score[1])

    # 5). Plotting decision boundary
    blobs.plot_decision_boundary(model, X, Y)

    # 6). Save model
    blobs.saveModel(model, './models/blobs_data_model.keras')

# ------------------------------------------------- #
    

# --------------------- Circles --------------------- #
    
    # circles = Classifying("circles")

    # data_circles = circles.generateData()
    # circles_x = data_circles.drop('cluster', axis=1)
    # circles_y = data_circles['cluster']

    # # 1). Prepare Data
    # X = circles.preparing_data(circles_x)
    # Y = circles_y.values.reshape(-1, 1) # transpose to resize

    # # 2). Build Model
    # model_circles = circles.buildModel()

    # # 3). Complie and Train Model
    # model, x_test, y_test = circles.trainingModel(model_circles, X, Y)

    # # 4). Evaluation and Testing Model
    # score = circles.testModel(model, x_test, y_test)
    # print("Loss: ", score[0])
    # print("Accuracy: ", score[1])

    # # 5). Plotting decision boundary
    # circles.plot_decision_boundary(model, X, Y)

    # # 6). Save model
    # circles.saveModel(model, './models/circles_data_model.keras')

# ------------------------------------------------- #


# --------------------- Moons --------------------- #
    
    # moons = Classifying("moons")

    # data_moons = moons.generateData()
    # moons_x = data_moons.drop('cluster', axis=1)
    # moons_y = data_moons['cluster']

    # # 1). Prepare Data
    # X = moons.preparing_data(moons_x)
    # Y = moons_y.values.reshape(-1, 1) # transpose to resize

    # # 2). Build Model
    # model_moons = moons.buildModel()

    # # 3). Complie and Train Model
    # model, x_test, y_test = moons.trainingModel(model_moons, X, Y)

    # # 4). Evaluation and Testing Model
    # score = moons.testModel(model, x_test, y_test)
    # print("Loss: ", score[0])
    # print("Accuracy: ", score[1])

    # # 5). Plotting decision boundary
    # moons.plot_decision_boundary(model, X, Y)

    # # 6). Save model
    # moons.saveModel(model, './models/moons_data_model.keras')

# ------------------------------------------------- #