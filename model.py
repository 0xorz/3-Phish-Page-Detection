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


def get_my_pecision_recall(clt, X, y):
    random_state = np.random.RandomState(42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25 , random_state=random_state)

    clt =clt.fit(X_train,y_train)

    probas_ = clt.predict_proba(X_test)

    precision, recall, _ = precision_recall_curve(y_test, probas_[:, 1])

    auc_pr = auc(recall, precision)

    print ( "precision score :%f"  %auc_pr)
    return precision, recall, auc_pr


def precision_recall_curve_draw(X,y):
    #KNN
    knn = KNeighborsClassifier(algorithm='auto', leaf_size=30,
                               metric='minkowski', n_neighbors=5, p=2, weights='uniform')
    #decision tree
    #dtree = DecisionTreeClassifier( criterion='entropy', min_samples_leaf=4, min_samples_split=5,
    #                                random_state=None, splitter='best')
    svmrbf= svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,  kernel='rbf',
                    max_iter=-1, probability=True, random_state=None,
                    shrinking=True, tol=0.001, verbose=False)
    #random forest
    rforest = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=None, max_features='auto', class_weight='balanced',
                                     min_samples_leaf=4, min_samples_split=5, n_estimators=10, n_jobs=1, oob_score=False, random_state=3)


    p_knn, r_knn, auc_knn = get_my_pecision_recall(knn, X, y)
    #p_dtree, r_dtree, auc_dtree = get_my_pecision_recall(dtree, X, y)
    p_rforest, r_rforest, auc_rforest = get_my_pecision_recall(rforest, X, y)
    p_svmrbf, r_svmrbf, auc_svmrbf = get_my_pecision_recall(svmrbf, X, y)

    """
    plt.clf()
    plt.plot(r_svmrbf, p_svmrbf, 'y.--', label='SVM auc=%0.3f' % auc_svmrbf)
    plt.plot(r_knn, p_knn, 'r^--', label='KNN auc=%0.3f' %auc_knn)

    #plt.plot(r_dtree, p_dtree, 'b>--', label ='Decision Tree auc=%0.3f'% auc_dtree)
    plt.plot(r_rforest, p_rforest, 'go--', label ='RF auc=%0.3f'% auc_rforest)

    plt.grid()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('recall rate')
    plt.ylabel('precision rate')
    plt.title('precision-recall curve')
    plt.legend(loc="lower right")
    plt.show()

    del X
    del y
    """


def get_fpr_tpr(clt, x, y):
    print ("\n")
    print (clt)

    random_state = np.random.RandomState(0)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=random_state)

    clt = clt.fit(X_train, y_train)
    y_pred = clt.predict(X_test)


    #accuracy score
    _accuracy_score = accuracy_score(y_test, y_pred)
    print ("Accuracy score {}".format(_accuracy_score))

    pred = clt.predict(X_test)
    fp, fn = 0, 0
    for i,c in enumerate(pred):
        if c == 1 and y_test[i] == 0:
            fp += 1
        if c == 0 and y_test[i] == 1:
            fn += 1
    print ("False positive: %f" % (float(fp + 0.0)/len(y_test)))
    print ("False negative: %f" % (float(fn + 0.0) / len(y_test)))

    #roc curve
    probas_ = clt.predict_proba(X_test)
    #print (probas_)
    #draw_confusion_matrix(y_test,y_pred)

    #print probas_
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    #print (fpr, tpr,thresholds)
    roc_auc = auc(fpr, tpr)
    print ("Area under the ROC curve : %f" % roc_auc)
    return fpr, tpr, roc_auc


def draw_confuse_matrix(x, y, clt=None):
    if clt is None:
        clt = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=None, max_features='auto',
                                         class_weight='balanced',
                                         min_samples_leaf=1, min_samples_split=2, n_estimators=50, n_jobs=1,
                                         oob_score=False, random_state=3)
    print (clt)

    random_state = np.random.RandomState(0)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=random_state)

    clt = clt.fit(X_train, y_train)
    y_pred = clt.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    print(cm)

    # Show confusion matrix in a separate window
    """
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    """


def train_and_draw_roc(X_original, y):

    bayes = GaussianNB()
    #KNN
    knn = KNeighborsClassifier(algorithm='auto', leaf_size=30,
                               metric='minkowski', n_neighbors=5, p=2, weights='uniform')

    #decision tree
    #dtree = DecisionTreeClassifier( criterion='entropy', min_samples_leaf=4, min_samples_split=5,
    #                                random_state=None, splitter='best')

    svmrbf= svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,  kernel='rbf',
                    max_iter=-1, probability=True, random_state=None,
                    shrinking=True, tol=0.001, verbose=False)

    rforest = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=None, max_features='auto', class_weight='balanced',
                                     min_samples_leaf=5, min_samples_split=5, n_estimators=50, n_jobs=1, oob_score=False, random_state=42)

    logit = linear_model.LogisticRegression()

    def print_fpr_tpr(fpr_knn, tpr_knn):
        for i, j in zip(fpr_knn, tpr_knn):
            print (str(i)+","+str(j))

    X = np.asarray(X_original)
    print ("Train shape {}".format(X.shape))
    print ("1-label: {}".format(sum(1 for i in Y if i==1)))

    print ("KNN")
    get_scroe_using_cv(knn, X, y)

    #print ("DT")
    #get_scroe_using_cv(dtree, X, y)

    print ("RF")
    get_scroe_using_cv(rforest, X, y)

    #print ("SVM")
    #get_scroe_using_cv(svmrbf, X, y)

    print ("Logit")
    get_scroe_using_cv(logit, X, y)

    fpr_knn, tpr_knn, auc_knn = get_fpr_tpr(knn, X, y)

    print ("=============KNN================")
    #print_fpr_tpr(fpr_knn,tpr_knn)

    print ("=============================")
    #fpr_dtree, tpr_dtree, auc_dtree = get_fpr_tpr(dtree, X, y)
    fpr_rforest, tpr_rforest, auc_rforest = get_fpr_tpr(rforest, X, y)

    print ("=============Forest================")
    print_fpr_tpr(fpr_rforest, tpr_rforest)

    print ("=============================")

    fpr_nb, tpr_nb, auc_nb = get_fpr_tpr(bayes, X, y)

    print ("=============NB================")
    #print_fpr_tpr(fpr_nb, tpr_nb)



def train_and_draw_roc_for_different_set_features(X_original, y):
    #random forest
    rforest = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=None, max_features='auto', class_weight='balanced',
                                     min_samples_leaf=5, min_samples_split=5, n_estimators=50, n_jobs=1, oob_score=False, random_state=42)

    X = np.asarray(X_original)
    X1 = np.delete(X, np.s_[988:], axis=1)  # only image

    X2 = np.delete(X, np.s_[0:988], axis=1)  # only text
    X2 = np.delete(X2, np.s_[988:], axis=1)

    X3 = np.delete(X, np.s_[0:1976], axis=1) # only form

    print (np.array_equal(X1,X2))
    print ("only image", X1.shape)
    print ("only text", X2.shape)
    print ("only form", X3.shape)


    print ("TOTAL Train shape {}".format(X.shape))
    print ("1-label: {}".format(sum(1 for i in Y if i==1)))

    #get_scroe_using_cv(dtree, X, y)
    #plt.clf()

    Xs = [X, X1, X2, X3]
    color = ['b', 'g', 'r', 'y']
    l = ['All','Image Only','Text only','Form only']
    print ("RF")
    i = 0
    for Xx,c in zip(Xs,color):

        print ("============={}=============".format(l[i]))
        print ("Train shape {}".format(Xx.shape))
        pca = decomposition.PCA(n_components=100)
        pca.fit(Xx)
        Xx = pca.transform(Xx)
        get_scroe_using_cv(rforest, Xx, y)
        fpr_rforest, tpr_rforest, auc_rforest = get_fpr_tpr(rforest, Xx, y)
        print ("===================\n\n")
        s = " %.3f" % auc_rforest
        label = l[i] + s
        #plt.plot(fpr_rforest, tpr_rforest, c+'o--', label = label)
        i += 1

    """
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('FPR(False Positive Rate)',fontsize=20)
    plt.ylabel('TPR(True Positive Rate)',fontsize=20)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.grid()
    plt.show()

    del X
    del y
    """


def train_tree(x, y):
    x = np.asarray(x)

    # random forest parameters
    if forest is None:
        # random forest
        forest = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=None, max_features='auto',
                                         class_weight='balanced',
                                         min_samples_leaf=5, min_samples_split=5, n_estimators=50, n_jobs=1,
                                         oob_score=False, random_state=42)
    get_scroe_using_cv(forest, x, y)
    forest.fit(x, y)

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")

    for f in range(x.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    """
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(x.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(x.shape[1]), indices)
    plt.xlim([-1, x.shape[1]])
    plt.show()
    """
    return forest


if __name__ == "__main__":
    pass
