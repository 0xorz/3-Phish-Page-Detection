import numpy as np
from sklearn import decomposition
from sklearn.externals import joblib

import model


def train():
    X = np.loadtxt("data/X.txt")
    Y = np.loadtxt("data/Y.txt")
    print('X shape', X.shape)
    print('Y shape', Y.shape)

    pca = decomposition.PCA(n_components=100)
    pca.fit(X)
    X = pca.transform(X)
    print ("X shape after PCA", X.shape)

    forest = model.train_tree(X, Y)
    
    joblib.dump(forest, 'saved_models/forest_pca.pkl')

    return forest


if __name__ == '__main__':
    train()