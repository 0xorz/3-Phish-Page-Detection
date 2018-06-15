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

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    joblib.dump(forest, 'saved_models/forest_pca.pkl')

    return forest


if __name__ == '__main__':
    train()