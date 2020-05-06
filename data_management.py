import pandas as pd
import sklearn.model_selection
import numpy as np


def data_loading():
    data = pd.read_csv("input_data/data.csv", delimiter=',',
                       index_col=None, encoding='utf8')
    data[['y']] = data[['y']].values - 1

    data_mapping = pd.read_csv("input_data/categories_mapping.csv",
                               delimiter=',', index_col=None, encoding='utf8')

    data = pd.merge(data, data_mapping, left_on='y', right_on='y', how='inner')
    data = data.set_index('Index')

    return data


def dimensionality_reduction(data, reduction_type):
    reduction_model = None

    if reduction_type == "PCA":
        reduction_model = sklearn.decomposition.PCA(n_components=2,
                                                    svd_solver='full')
    elif reduction_type == "FA":
        reduction_model = \
            sklearn.decomposition.FactorAnalysis(n_components=2,
                                                 tol=1e-5,
                                                 max_iter=10000,
                                                 svd_method='lapack')
    elif reduction_type == "MDS":
        reduction_model = sklearn.manifold.MDS()
    elif reduction_type == "TSNE":
        reduction_model = sklearn.manifold.TSNE(perplexity=5, n_iter=2500)

    return pd.DataFrame(np.append(reduction_model.fit_transform(
        data.iloc[:, 0:178]), data.iloc[:, 178:180].values, axis=1),
        columns=['x1', 'x2', 'y', 'categories'])


def categories_preprocessing(data):
    return


def sets_creation(data_input, data_output):
    training_set_input, validation_set_input, training_set_output, \
        validation_set_output = \
        sklearn.model_selection.train_test_split(data_input, data_output,
                                                 test_size=0.2)

    return (training_set_input, training_set_output), \
           (validation_set_input, validation_set_output)

