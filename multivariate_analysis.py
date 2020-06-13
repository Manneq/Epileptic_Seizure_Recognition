"""
    File 'multivariate_analysis.py' used for multivariate analysis.
"""
import pandas as pd
import sklearn.metrics
import sklearn.ensemble
import numpy as np
import data_management
import plotting
import neural_network


def standard_deviation_computing(data,
                                 data_type="Initial"):
    """
        Method to compute and plot standard deviations of parameters.
        param:
            1. data - pandas DataFrame of initial or normalized data
            2. data_type - string value of type of dataset
                ('Initial' as default)
    """
    # Standard deviations plotting
    plotting.histogram_plotting(np.std(data, axis=0),
                                ["Parameters", "Standard deviation"],
                                data_type + " data standard deviations",
                                "multivariate_analysis/initial/"
                                "standard_deviations")

    # Standard deviations for categories plotting
    for category in np.unique(data[['categories']].values):
        plotting.histogram_plotting(np.std(data.loc[
                                               data['categories'] == category],
                                           axis=0),
                                    ["Parameters", "Standard deviation"],
                                    data_type +
                                    " data standard deviations (" +
                                    category + ")",
                                    "multivariate_analysis/initial/"
                                    "standard_deviations")

    return


def random_forrest_classification(training_set, validation_set, folder,
                                  feature_importance=True):
    """
        Training and validation of random forrest classifier model.
        param:
            1. training_set - list of sets for training
            2. validation_set - list of sets for validation
            3. folder - string name of folder
            4. feature_importance - boolean value to compute most important
                features (True as default)
        return:
            most_important_features - list of 15 most important features
    """
    # Creation and training of random forest model
    model_random_forest = \
        sklearn.ensemble.RandomForestClassifier(n_estimators=2500,
                                                max_features=None,
                                                n_jobs=-1,
                                                random_state=0).\
        fit(training_set[0],
            training_set[1].drop(columns=['categories'], axis=1).
            astype('int').values.ravel())

    categories = validation_set[1].loc[:, ['y', 'categories']].\
        sort_values(by=['y']).drop(columns=['y'])['categories'].unique()

    # Accuracy computation for every category
    confusion_matrix = sklearn.metrics.confusion_matrix(
        validation_set[1].drop(columns=['categories'], axis=1).astype('int'),
        model_random_forest.predict(validation_set[0]),
        labels=np.arange(0, len(categories)))

    accuracies = (confusion_matrix.astype('float') /
                  confusion_matrix.sum(axis=1)[:, np.newaxis]).diagonal()

    # Precision, recall and F-score computation for every category
    precisions, recalls, f_scores, _ = \
        sklearn.metrics.precision_recall_fscore_support(
            validation_set[1].drop(columns=['categories'], axis=1).
            astype('int'),
            model_random_forest.predict(validation_set[0]),
            labels=np.arange(0, len(categories)))

    # Accuracy distribution over categories plotting
    plotting.bar_plotting(pd.Series(accuracies, index=categories).
                          sort_values(ascending=False),
                          ["Categories", "Accuracy"],
                          "Classification accuracy",
                          "multivariate_analysis/initial/random_forest/" +
                          folder)

    # Recall distribution over categories plotting
    plotting.bar_plotting(pd.Series(recalls, index=categories).
                          sort_values(ascending=False),
                          ["Categories", "Recall"],
                          "Classification recall",
                          "multivariate_analysis/initial/random_forest/" +
                          folder)

    # Precision distribution over categories plotting
    plotting.bar_plotting(pd.Series(precisions, index=categories).
                          sort_values(ascending=False),
                          ["Categories", "Precision"],
                          "Classification precision",
                          "multivariate_analysis/initial/random_forest/" +
                          folder)

    # F-score distribution over categories plotting
    plotting.bar_plotting(pd.Series(f_scores, index=categories).
                          sort_values(ascending=False),
                          ["Categories", "F-score"],
                          "Classification F-score",
                          "multivariate_analysis/initial/random_forest/" +
                          folder)

    # Decision tree plotting
    plotting.graph_exporting(model_random_forest,
                             validation_set[0].columns.tolist(),
                             validation_set[1]['y'].unique().astype('str'),
                             folder)

    most_important_features = None

    # Most important features selection and plotting
    if feature_importance:
        most_important_features = \
            pd.Series(model_random_forest.feature_importances_,
                      index=validation_set[0].columns).\
            sort_values(ascending=False).head(15)

        plotting.bar_plotting(most_important_features,
                              ["Brain activity", "Score"],
                              "Top 15 most important features",
                              "multivariate_analysis/initial/random_forest/" +
                              folder)

    return most_important_features


def classification_task(training_set, validation_set):
    """
        Function to perform classification task using random forest
            with different approaches.
        param:
            1. training_set - list of sets for training
            2. validation_set - list of sets for validation
    """
    # Random forest model on training and validation sets
    most_important_features = \
        random_forrest_classification(training_set, validation_set, "full")

    # Random forest model on training and validation sets with only 15
    # most important features
    random_forrest_classification([training_set[0].loc[:,
                                   most_important_features.index],
                                   training_set[1]],
                                  [validation_set[0].loc[:,
                                   most_important_features.index],
                                   validation_set[1]],
                                  "full_featured", feature_importance=False)

    # Sets conversion for seizure and seizure classification
    training_set_seizure, validation_set_seizure, \
        training_set_not_seizure, validation_set_not_seizure = \
        data_management.sets_conversion(training_set, validation_set)

    # Random forest model on training and validation sets for seizure
    # classification
    most_important_features = \
        random_forrest_classification(training_set_seizure,
                                      validation_set_seizure, "seizure")

    # Random forest model on training and validation sets with only 15
    # most important features for seizure classification
    random_forrest_classification([training_set_seizure[0].loc[:,
                                   most_important_features.index],
                                   training_set_seizure[1]],
                                  [validation_set_seizure[0].loc[:,
                                   most_important_features.index],
                                   validation_set_seizure[1]],
                                  "seizure_featured", feature_importance=False)

    # Random forest model on training and validation sets for not seizure
    # classification
    most_important_features = \
        random_forrest_classification(training_set_not_seizure,
                                      validation_set_not_seizure,
                                      "not_seizure")

    # Random forest model on training and validation sets with only 15
    # most important features for not seizure classification
    random_forrest_classification([training_set_not_seizure[0].loc[:,
                                   most_important_features.index],
                                   training_set_not_seizure[1]],
                                  [validation_set_not_seizure[0].loc[:,
                                   most_important_features.index],
                                   validation_set_not_seizure[1]],
                                  "not_seizure_featured",
                                  feature_importance=False)

    return


def initial_data_analysis(data):
    """
        Method to analyse initial dataset.
        param:
            data - pandas DataFrame of initial data
    """
    # Correlations plotting
    plotting.heatmap_plotting(data.iloc[:, 0:178].corr(),
                              "Brain activity correlations",
                              "initial", width=3000, height=3000,
                              annotation=False)

    # Standard deviations computing
    standard_deviation_computing(data)

    # Data normalization
    data = data_management.data_normalization(data)

    # Standard deviations computing for normalized data
    standard_deviation_computing(data, data_type="Normalized")

    # Training and validation sets creation for decision tree
    training_set, validation_set = data_management.sets_creation(data)

    # Random forest for classification problem
    classification_task(training_set, validation_set)

    # Training and validation sets creation for neural network
    training_set, validation_set = \
        data_management.sets_creation(data_management.data_preprocessing(data))

    # Neural network for classification problem
    neural_network.neural_network_model(training_set, validation_set)

    return


def reduced_data_analysis(data,
                          reduction_type="PCA"):
    """
        Method for data with reduced dimensions analysis.
        param:
            1. data - pandas DataFrame of initial data.
            2. reduction_type - string value of reduction algorithm
                ('PCA' as default)
    """
    # Dimensionality reduction
    data = data_management.dimensionality_reduction(data, reduction_type)

    # Data plotting
    plotting.data_plotting(data,
                           ["Brain activity X1", "Brain activity X2"],
                           reduction_type + " brain activity", reduction_type)

    # Correlations plotting
    plotting.heatmap_plotting(data.iloc[:, 0:2].corr(),
                              reduction_type + " brain activity correlations",
                              reduction_type)

    return


def multivariate_analysis_applying(data):
    """
        Method to apply multivariate analysis.
        param:
            data - pandas DataFrame of initial data
    """
    # Initial data analysis
    initial_data_analysis(data)

    # Data with reduced dimensions analysis
    reduced_data_analysis(data)

    reduced_data_analysis(data, reduction_type="FA")

    reduced_data_analysis(data, reduction_type="MDS")

    reduced_data_analysis(data, reduction_type="TSNE")

    return
