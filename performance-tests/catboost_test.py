#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 01 09:44:30 2018

@author: alex
"""

from pre import X_train, X_test, y_train, y_test
from catboost import Pool, CatBoost
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score
from utils import rand_search_result, print_dict


def basic_test():
    cat_features = [0, 1, 6]
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool = Pool(X_test, cat_features=cat_features)

    param = {'iterations':10, 'depth':2, 'learning_rate':1, 'loss_function':'Logloss'}
    model = CatBoost(param)

    model.fit(train_pool)

    preds_class = model.predict(test_pool, prediction_type='Class')

    eq = y_test == np.array(preds_class)
    print(eq.sum() / eq.shape[0])


def adaptive_test(X_train, X_test, y_train, y_test, n_iter=100, n_jobs=4, cv=3):
    cat_features = [0, 1, 6]

    classifier = CatBoost()
    auc = make_scorer(roc_auc_score)

    rand_list = {'iterations': [1, 2, 5, 10, 20, 50, 100, 200],
                 'depth': [*range(1, 11)],
                 'learning_rate': [.005, .01, .02, .05, .1, .2, .5, 1.0, 2.0, 5.0, 10.0],
                 'loss_function': ['Logloss']}

    rand_search = RandomizedSearchCV(classifier, param_distributions=rand_list, n_iter=n_iter, n_jobs=n_jobs, cv=cv, random_state=2017, scoring=auc)

    rand_search.fit(X_train, y_train)

    y_pre = rand_search.best_estimator_.predict(X_test, prediction_type='Class')

    return rand_search_result(rand_search.best_params_, y_test, y_pre)


if __name__ == '__main__':
    print_dict(adaptive_test(X_train, X_test, y_train, y_test, ))

