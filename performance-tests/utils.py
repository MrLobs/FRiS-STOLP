from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def rand_search_result(best_params, y_test, y_pre):
    return {
        'best_params': best_params,
        'precision': precision_score(y_test, y_pre),
        'recall': recall_score(y_test, y_pre),
        'accuracy': accuracy_score(y_test, y_pre),
        'f1_score': f1_score(y_test, y_pre)
    }


def print_dict(container):
    for x in container.items():
        print(x)


def make_test_data(features, labels, do_pca = False, n_components = 3):
    data = PCA(n_components=3).fit_transform(features) if do_pca else features
    return train_test_split(data, labels, test_size=0.2, random_state=42)
