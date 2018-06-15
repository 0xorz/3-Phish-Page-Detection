#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np

from sklearn.model_selection import *

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

from sklearn import decomposition

from sklearn.naive_bayes import GaussianNB


# this is to get score using cross_validation
def get_scroe_using_cv(clt, X, y):
    scores = cross_val_score(clt, X, y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def train_tree(x, y):
    x = np.asarray(x)

    # random forest
    forest = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=None, max_features='auto',
                                        class_weight='balanced',
                                        min_samples_leaf=5, min_samples_split=5, n_estimators=50, n_jobs=1,
                                        oob_score=False, random_state=42)
    get_scroe_using_cv(forest, x, y)
    forest.fit(x, y)
    
    return forest


if __name__ == "__main__":
    pass
