from fs import fris_stolp
from pre import *
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, f1_score, recall_score
from scipy import stats
from utils import rand_search_result, print_dict
from sklearn.base import BaseEstimator, ClassifierMixin


metric = f1_score


class SklearnHelper(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.5, v=2):
        """
        :param threshold: (-1 .. 1)
        :param v: {1, 2}
        """
        self.threshold = threshold
        self.v = v

    def get_params(self, deep=False):
        return {"threshold" : self.threshold, "v" : self.v}#self.params

    def set_params(self, **parameters):
        #params_src = params if len(kwargs) == 0 else kwargs
        self.threshold, self.v = parameters["threshold"], parameters["v"]
        self.clf = fris_stolp(self.threshold, self.v)
        return self

    def train(self, x, y):
        self.clf.fit(x, np.int8(y))

    def predict(self, x):
        res = self.clf.predict(x)
        return res

    def fit(self, x, y):
        return self.clf.fit(x, np.int8(y))

    def feature_importances(self, x, y):
        pass


def basic_test():
    classifier = fris_stolp(.5, 1)
    classifier.fit(X_train.values, np.int8(y_train.values))
    res = classifier.predict(X_test.values)
    print(res)
    return rand_search_result({}, y_test, res)


def adaptive_test(X_train, X_test, y_train, y_test, n_iter=50, n_jobs=4, cv=2):
    classifier = SklearnHelper()

    auc = make_scorer(roc_auc_score)

    rand_list = {"v": [1],
                 "threshold": stats.uniform(-.99, 1.99)}

    rand_search = RandomizedSearchCV(classifier, param_distributions=rand_list, n_iter=n_iter, n_jobs=n_jobs,
                                     cv=cv, random_state=42, scoring=auc)
    rand_search.fit(X_train, np.int8(y_train))

    y_pre = rand_search.predict(X_test)

    return rand_search_result(rand_search.best_params_, y_test, y_pre)


if __name__ == '__main__':
    #clone(fris_stolp())
    from utils import get_iris
    # X_train, X_test, y_train, y_test = get_iris()
    # y_train[y_train == 2] = 1
    # y_test[y_test == 2] = 1
    print_dict(adaptive_test(X_train, X_test, y_train, y_test, cv=2, n_iter=50))
