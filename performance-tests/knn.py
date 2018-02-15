from pre import *
from sklearn import neighbors
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score
from utils import rand_search_result, print_dict


def adaptive_test(X_train, X_test, y_train, y_test, n_iter=100, n_jobs=4, cv=3):
    classifier = neighbors.KNeighborsClassifier()

    auc = make_scorer(roc_auc_score)

    rand_list = {"n_neighbors": [4,5,6,7,8,9],
                 "leaf_size": [25, 30, 35],
                 "algorithm" : ['ball_tree', 'kd_tree', 'brute'],
                 "p" : [1,2]}

    rand_search = RandomizedSearchCV(classifier, param_distributions=rand_list,
                                     n_iter=n_iter, n_jobs=n_jobs, cv=cv, random_state=2017, scoring=auc)
    rand_search.fit(X_train, y_train)

    y_pre = rand_search.predict(X_test)

    return rand_search_result(rand_search.best_params_, y_test, y_pre)


if __name__ == '__main__':
    print_dict(adaptive_test(X_train, X_test, y_train, y_test, ))
