#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

### your code goes here

from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)


def make_classifer():
    clf = DecisionTreeClassifier()
    clf.fit(features_train, labels_train)
    return clf

clf = make_classifer()
pred = clf.predict(features_test)

def num_poi(vs):
    return len(
        filter(
            lambda v: v > 0,
            vs
        )
    )

def true_positives(labels, pred):
    return filter(
        lambda (l, p): l == p == 1.0,
        zip(labels, pred)
    )

def score(labels, pred):
    return {
        'precision_score': precision_score(labels, pred),
        'recall_score': precision_score(labels, pred),
        'accuracy': accuracy_score(labels, pred)
    }
