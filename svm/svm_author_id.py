#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(kernel='rbf', C=10000)

def pred():
    clf.fit(features_train, labels_train)
    return clf.predict(features_test)

def accuracy():
    return accuracy_score(labels_test, pred())

def specific_scores():
    predictions = pred()
    return [
        '10:', predictions[10], '26:', predictions[26], '50:', predictions[50]
    ]

def by_chris():
    return len(filter(lambda x: x == 1, pred()))

def perf():
    def f(label, f):
        t0 = time()
        f()
        print label , " : " , round(time()-t0, 3), "s"

    f("training", lambda : clf.fit(features_train, labels_train))
    f("predicting", lambda : clf.predict(features_test))

print by_chris()

#########################################################
