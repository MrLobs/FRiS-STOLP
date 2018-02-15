from pre import *
from xgboost import XGBClassifier
from sklearn.model_selection import  RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score
from scipy import stats
from utils import rand_search_result, print_dict


def adaptive_test(X_train, X_test, y_train, y_test, n_iter=100, n_jobs=4, cv=3):
    classifier = XGBClassifier()

    auc = make_scorer(roc_auc_score)

    rand_list = {"learning_rate": stats.uniform(1e-6, 1e-2),
                 "n_estimators" : [100, 300, 500]}


    rand_search = RandomizedSearchCV(classifier, param_distributions=rand_list,
                                     n_iter=n_iter, n_jobs=n_jobs, cv=cv, random_state=2017, scoring=auc)
    rand_search.fit(X_train, y_train)

    y_pre = rand_search.predict(X_test)

    return rand_search_result(rand_search.best_params_, y_test, y_pre)


if __name__ == '__main__':
    print_dict(adaptive_test(X_train, X_test, y_train, y_test, ))
