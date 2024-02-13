'''
Assignment  : Titanic Survival Challenge
Date        : 12 Feb 2024
Name        : Chonakan Chumtap 
Student ID  : 6410301022
'''

from sklearn import metrics


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
        else :
            model = Sequential([
                layers.Dense(8, activation="relu", input_shape=(3,)),
                layers.Dense(4, activation="relu"),
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
        else :
            model.compile(loss='mean_squared_error',
                        optimizer='adam',
                        metrics=['accuracy'])
            
            X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=0.5, random_state=1)

            start = time.time()
            history = model.fit(X_train, Y_train, epochs=150, verbose=1, batch_size=25, validation_split=0.25)
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
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from keras import Sequential
    from keras import layers

    url = "titanic.csv"
    data = pd.read_csv(url)

    titanic = Classifying('none')

    # 1). Prepare Data
    data_X = data[['Fare', 'Age']].values
    data_y = data['Survived'].values
    data_F3 = np.zeros((len(data_X),), dtype=int)

    for i in range(len(data_X)):
        if np.isnan(data_X[i]).any():
            data_F3[i] = 1
    
    data_X[np.isnan(data_X)] = np.nanmedian(data_X)
    data_X_with_F3 = np.column_stack((data_X, data_F3))

    X = titanic.preparing_data(data_X_with_F3)
    y = data_y.reshape(-1, 1)

    # 2). Build Model
    model_titanic = titanic.buildModel()

    # 3). Complie and Train Model
    model, x_test, y_test = titanic.trainingModel(model_titanic, X, y)

    # 4). Evaluation and Testing Model
    score = titanic.testModel(model, x_test, y_test)
    print("Loss: ", score[0])
    print("Accuracy: ", score[1])

    y_pred_prob = model.predict(x_test)
    y_pred = np.where(y_pred_prob>0.5, 1, 0)

    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_prob)
    auc = metrics.roc_auc_score(y_test, y_pred_prob)

    plt.plot(fpr, tpr, label="AUC="+str(auc))
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.legend(loc=4)
    plt.show()