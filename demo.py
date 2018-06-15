#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn import decomposition
from sklearn.externals import joblib

import model
import util
import feature_extract


try:
    import cPickle as pickle
except:
    print ("No cpickle? use pickle")
    import pickle


def predict(dire):
    X = np.loadtxt("./data/X.txt")
    print (X.shape)

    pca = decomposition.PCA(n_components=100)
    pca.fit(X)

    print ("PCA fitted")

    forest = None

    try:
        forest = joblib.load('./saved_models/forest_pca.pkl')
    except:

        print ("Existing model cannot be used, maybe the sklearn version problem?")
        print ("We begin to retrain the model")
        X = np.loadtxt("./data/X.txt")
        Y = np.loadtxt("./data/Y.txt")
        print ("X shape", X.shape)

        pca2 = decomposition.PCA(n_components=100)
        pca2.fit(X)
        X = pca2.transform(X)
        print ("X shape after PCA", X.shape)

        forest = model.tree_model_train_and_save(X, Y)

    candidates = util.read_crawl_candidates(dire)

    for cand in candidates:
        idx = cand.get_idx()
        v = feature_extract.feature_vector_extraction(cand)

        if not v:
            print ("Fail to extract feature vectors of {}".format(idx))
            continue

        new_v = pca.transform(np.asarray(v).reshape(1, -1))
        p_prob = forest.predict_proba(new_v)
        p = forest.predict(new_v)
        print ("Prediction result:\t", str(idx) + "----" + str(p.tolist()[0]) + "----" + str(p_prob.tolist()[0]))

    return


if __name__ == "__main__":
    dire = os.getcwd() + "/test/"
    print (dire)
    predict(dire)

