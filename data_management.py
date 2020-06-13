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
            2 lists of sets for training and validation
    """
    # Sets creation with 20% for validation
    training_set_input, validation_set_input, training_set_output, \
        validation_set_output = \
        sklearn.model_selection.train_test_split(data.iloc[:, 0:178],
                                                 data.iloc[:,
                                                 178:data.shape[1]],
                                                 test_size=0.2)

    return [training_set_input, training_set_output], \
           [validation_set_input, validation_set_output]


def sets_conversion(training_set, validation_set):
    """
        Training and validation sets conversion into seizure and
            not seizure sets.
        param:
            1. training_set - list of sets for training
            2. validation_set - list of sets for validation
        return:
            1. seizure_training_set - list of sets for training seizure
                classification
            2. seizure_validation_set - list of sets for seizure
                classification validation
            3. not_seizure_training_set - list of sets for training not
                seizure classification
            4. not_seizure_validation_set - list of sets for not seizure
                classification validation
    """
    # Training and validation sets for seizure classification creation
    seizure_training_set = \
        [pd.DataFrame(training_set[0].values, index=training_set[0].index,
                      columns=training_set[0].columns),
         pd.DataFrame(training_set[1].values, index=training_set[1].index,
                      columns=training_set[1].columns)]
    seizure_validation_set = \
        [pd.DataFrame(validation_set[0].values, index=validation_set[0].index,
                      columns=validation_set[0].columns),
         pd.DataFrame(validation_set[1].values, index=validation_set[1].index,
                      columns=validation_set[1].columns)]

    seizure_training_set[1].loc[seizure_training_set[1]['y'] != 0,
                                ['y', 'categories']] = 1, 'Not seizure'
    seizure_validation_set[1].loc[seizure_validation_set[1]['y'] != 0,
                                  ['y', 'categories']] = 1, 'Not seizure'

    # Training and validation sets for not seizure classification creation
    not_seizure_training_set = pd.merge(training_set[0],
                                        training_set[1],
                                        left_index=True, right_index=True)
    not_seizure_validation_set = pd.merge(validation_set[0],
                                          validation_set[1],
                                          left_index=True, right_index=True)

    not_seizure_training_set = \
        not_seizure_training_set.loc[not_seizure_training_set['y'] != 0]
    not_seizure_validation_set = \
        not_seizure_validation_set.loc[not_seizure_validation_set['y'] != 0]

    not_seizure_training_set.loc[not_seizure_training_set['y'] != 0,
                                 ['y']] -= 1
    not_seizure_validation_set.loc[not_seizure_validation_set['y'] != 0,
                                   ['y']] -= 1

    not_seizure_training_set = [not_seizure_training_set.iloc[:, 0:178],
                                not_seizure_training_set.iloc[:,
                                178:not_seizure_training_set.shape[1]]]
    not_seizure_validation_set = [not_seizure_validation_set.iloc[:, 0:178],
                                  not_seizure_validation_set.iloc[:,
                                  178:not_seizure_validation_set.shape[1]]]

    return seizure_training_set, seizure_validation_set, \
        not_seizure_training_set, not_seizure_validation_set


def time_series_conversion(data):
    """
        Method to convert data to univariate time series.
        param:
            data - pandas DataFrame of initial data
        return:
            data - pandas DataFrame of time series for patients
    """
    # Categories dropping
    data = data.drop(columns=['y', 'categories'], axis=1)

    indexes = data.index
    time_data = np.empty((len(indexes), 2))

    # Time chunks and patients codes extraction
    for i in range(len(indexes)):
        temp_time = indexes[i].split(".")

        time_data[i, 0] = float(temp_time[0].replace('X', '')) - 1

        if temp_time[len(temp_time) - 1][0] == 'V':
            time_data[i, 1] = \
                int(temp_time[len(temp_time) - 1].replace('V', '')) + 10000
        else:
            time_data[i, 1] = int(temp_time[len(temp_time) - 1])

    data[['time', 'patients']] = pd.DataFrame(time_data, index=data.index)

    time_series_data = np.empty((len(data) * 178, 3))

    # Time series creation
    for i in range(len(data.index)):
        temp_data = data.iloc[i, :].values
        offset = 178 * i

        for j in range(len(temp_data) - 2):
            time_series_data[j + offset, :] = \
                [temp_data[179], temp_data[178] + j / 179, temp_data[j]]

    data = pd.DataFrame(time_series_data,
                        columns=['patients', 'time', 'brain_activity'])

    return data
