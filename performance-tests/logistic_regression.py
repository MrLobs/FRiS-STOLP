#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:38:39 2018

@author: rost
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:33:30 2018

@author: rost
"""
from pre import *
from sklearn import linear_model
from sklearn.model_selection import  RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score
from scipy import stats
from utils import rand_search_result, print_dict


def adaptive_test(X_train, X_test, y_train, y_test, n_iter=100, n_jobs=4, cv=3):
    classifier = linear_model.LogisticRegression()

    auc = make_scorer(roc_auc_score)

    rand_list = {"tol": stats.uniform(1e-6, 1e-4),
                 "C" : stats.uniform(0.5, 1),
                 "max_iter" : [100, 150, 200]}

    rand_search = RandomizedSearchCV(classifier, param_distributions=rand_list,
                                     n_iter=n_iter, n_jobs=n_jobs, cv=cv, random_state=2017, scoring=auc)
    rand_search.fit(X_train, y_train)

    y_pre = rand_search.predict(X_test)

    return rand_search_result(rand_search.best_params_, y_test, y_pre)


if __name__ == '__main__':
    print_dict(adaptive_test(X_train, X_test, y_train, y_test, ))
