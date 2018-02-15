#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 19:09:15 2018

@author: rost
"""

from pre import *
import svm
import decision_tree
import logistic_regression
import knn
import naive_bayes
import xg_boost
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools

import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from brew.base import Ensemble, EnsembleClassifier
from brew.stacking.stacker import EnsembleStack, EnsembleStackClassifier
from brew.combination.combiner import Combiner

from mlxtend.data import iris_data
from mlxtend.plotting import plot_decision_regions

# Initializing Classifiers
clf1 = LogisticRegression(random_state=0)
clf2 = knn.classifier
clf3 = xg_boost.classifier
import fris_stolp_test
clf4 = fris_stolp_test.SklearnHelper

# Creating Ensemble
ensemble = Ensemble([clf1, clf2, clf3, clf4])
eclf = EnsembleClassifier(ensemble=ensemble, combiner='mean')

# Creating Stacking
layer_1 = Ensemble([clf1, clf2, clf3])
layer_2 = Ensemble([sklearn.clone(clf1)])

stack = EnsembleStack(cv=3)

stack.add_layer(layer_1)
stack.add_layer(layer_2)

sclf = EnsembleStackClassifier(stack, combiner=Combiner('mean'))

sclf.fit(X_train.values, y_train.values)

y_pre = sclf.predict(X_test.values)

precision = precision_score(y_test, y_pre)
recall = recall_score(y_test, y_pre)
accuracy = accuracy_score(y_test, y_pre)
fmera = f1_score(y_test, y_pre)

if __name__ == '__main__':
    print("presicion ",precision, " recall ", recall," fmera ", fmera," accuracy ", accuracy)
