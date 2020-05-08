"""
    File 'data_management.py' used to load and preprocess dataset.
"""
import pandas as pd
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.model_selection
import numpy as np


def data_loading():
    """
        Method to load initial Raiffeisenbank clients transactions dataset.
        return:
            data - pandas DataFrame of epileptic seizures data
    """
    # Data loading
    data = pd.read_csv("input_data/data.csv", delimiter=',',
                       index_col=None, encoding='utf8')
    data[['y']] = data[['y']].values - 1

    # Mapping data loading
    data_mapping = pd.read_csv("input_data/categories_mapping.csv",
                               delimiter=',', index_col=None, encoding='utf8')

    # Mapping and data merging
    data = pd.merge(data, data_mapping, left_on='y', right_on='y', how='inner')
    data = data.set_index('Index')

    return data


def dimensionality_reduction(data, reduction_type):
    """
        Method for reducing dimensionality of data using:
            1. PCA
            2. FA
            3. MDS
            4. t-SNE
        param:
            1. data - pandas DataFrame of mapped data
            2. reduction_type - algorithm to use for reduction
        return:
            pandas DataFrame of data with 2-dimensional vector of
                parameters
    """
    reduction_model = None

    if reduction_type == "PCA":
        # PCA reduction model
        reduction_model = sklearn.decomposition.PCA(n_components=2,
                                                    svd_solver='full')
    elif reduction_type == "FA":
        # FA reduction model
        reduction_model = \
            sklearn.decomposition.FactorAnalysis(n_components=2,
                                                 tol=1e-5,
                                                 max_iter=10000,
                                                 svd_method='lapack')
    elif reduction_type == "MDS":
        # MDS reduction model
        reduction_model = sklearn.manifold.MDS()
    elif reduction_type == "TSNE":
        # t-SNE reduction model
        reduction_model = sklearn.manifold.TSNE(perplexity=5, n_iter=2500)

    return pd.DataFrame(np.append(reduction_model.fit_transform(
        data.iloc[:, 0:178]), data.iloc[:, 178:180].values, axis=1),
        columns=['x1', 'x2', 'y', 'categories'], index=data.index)


def data_normalization(data):
    """
        Method for data normalization.
        param:
            data - pandas DataFrame of mapped data
        return:
            pandas DataFrame of normalized data
    """
    # MinMax scaler model
    scaler_model = sklearn.preprocessing.MinMaxScaler()

    return pd.DataFrame(np.append(scaler_model.fit_transform(
        data.iloc[:, 0:178]), data.iloc[:, 178:180].values, axis=1),
        columns=data.columns, index=data.index)


def data_preprocessing(data):
    """
        Method for categories binarization.
        param
            data - pandas DataFrame of normalized data
        return:
            data - pandas DataFrame of normalized data with binarized
                categories
    """
    # Label binarizer model
    model_binarizer = sklearn.preprocessing.LabelBinarizer()

    # Categories binarization
    data = data.drop(columns=['y'], axis=1)
    data[['y1', 'y2', 'y3', 'y5', 'y6']] = \
        pd.DataFrame(model_binarizer.fit_transform(data.iloc[:, 178]),
                     index=data.index)

    return data


def sets_creation(data):
    """
        Method for creating training and validation sets.
        param:
            data - pandas DataFrame of normalized data
        return:
            2 tuples of sets for training and validation
    """
    # Sets creation with 20% for validation
    training_set_input, validation_set_input, training_set_output, \
        validation_set_output = \
        sklearn.model_selection.train_test_split(data.iloc[:, 0:178],
                                                 data.iloc[:,
                                                 178:data.shape[1]],
                                                 test_size=0.2)

    return (training_set_input, training_set_output), \
           (validation_set_input, validation_set_output)
