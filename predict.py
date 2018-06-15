import numpy as np
from sklearn import decomposition
from sklearn.externals import joblib

import feature_extract


def predict(img, html):
    X = np.loadtxt("data/X.txt")
    print (X.shape)

    pca = decomposition.PCA(n_components=100)
    pca.fit(X)
    
    forest = joblib.load('saved_models/forest_pca.pkl')

    v = feature_extract.extract_feature(img, html)
    new_v = pca.transform(np.asarray(v).reshape(1, -1))

    p_prob = forest.predict_proba(new_v)
    p = forest.predict(new_v)

    return [p.tolist()[0], p_prob.tolist()[0]]


if __name__ == "__main__":
    pass