#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:48:58 2018

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

from sklearn.ensemble import VotingClassifier
clf1 = logistic_regression.classifier
clf2 = xg_boost.classifier
clf3 = decision_tree.classifier

classifier = VotingClassifier(estimators=[('svm', clf1), ('deci', clf2), ('dei', clf3)], voting='soft')
classifier.fit(X_train, y_train)

y_pre = classifier.predict(X_test)

precision = precision_score(y_test, y_pre)
recall = recall_score(y_test, y_pre)
accuracy = accuracy_score(y_test, y_pre)
fmera = f1_score(y_test, y_pre)

print("presicion ",precision, " recall ", recall," fmera ", fmera," accuracy ", accuracy)