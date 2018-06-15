#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from sklearn import decomposition
from sklearn.externals import joblib

import model
import feature_extract

import sys
import os

# concurrent preidction
# from concurrent.futures import TimeoutError


def parse_options():
    parser = argparse.ArgumentParser(description="running analysis...", prefix_chars='-+/')
    parser.add_argument('-t', '--html', type=str,
                        help='A html source data to extract features')
    parser.add_argument('-i', '--img', type=str,
                        help='A image data to extract features')
    args = parser.parse_args()
    return args


def predict(img, html):
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

    v = feature_extract.feature_vector_extraction_from_img_html(img, html)
    if not v:
        print ("Fail to extract feature vectors.")
        return

    new_v = pca.transform(np.asarray(v).reshape(1, -1))
    p_prob = forest.predict_proba(new_v)
    p = forest.predict(new_v)
    print ("Prediction: ----" + str(p.tolist()[0]) + "----" + str(p_prob.tolist()[0]))

    return


def main():
    args = parse_options()

    img = os.path.abspath(args.img)
    html = os.path.abspath(args.html)

    print ("Run the prediction...")
    predict(img, html)

    return


if __name__ == "__main__":

    sys.exit(main())


