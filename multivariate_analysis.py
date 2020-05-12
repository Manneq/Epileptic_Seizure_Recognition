"""
    File 'multivariate_analysis.py' used for multivariate analysis.
"""
import pandas as pd
import numpy as np
import sklearn.metrics
import sklearn.tree
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
                                data_type + " data standard deviations")

    # Standard deviations for categories plotting
    for category in np.unique(data[['categories']].values):
        plotting.histogram_plotting(np.std(data.loc[
                                               data['categories'] == category],
                                           axis=0),
                                    ["Parameters", "Standard deviation"],
                                    data_type +
                                    " data standard deviations (" +
                                    category + ")")

    return


def classification_using_dt(training_set, validation_set):
    """
        Function to perform classification task using decision tree.
        param:
            1. training_set - tuple of sets for training
            2. validation_set - tuple of sets for validation
    """
    # Decision tree model
    decision_tree_model = \
        sklearn.tree.DecisionTreeClassifier(criterion='entropy',
                                            max_leaf_nodes=30).\
        fit(training_set[0],
            training_set[1].drop(columns=['categories'], axis=1).astype('int'))

    accuracies = []
    categories = validation_set[1]['categories'].unique().tolist()

    # Accuracy for every category computing
    for category in categories:
        accuracies.append(
            sklearn.metrics.accuracy_score(
                validation_set[1].loc[
                    validation_set[1]['categories'] == category].
                drop(columns=['categories'], axis=1).astype('int'),
                decision_tree_model.predict(validation_set[0].loc[
                    validation_set[1]['categories'] == category])))

    # Accuracy distribution over categories plotting
    plotting.bar_plotting(pd.Series(accuracies, index=categories).
                          sort_values(ascending=False),
                          ["Categories", "Accuracy"],
                          "Decision tree classification results (" +
                          str(sklearn.metrics.accuracy_score(
                              validation_set[1].drop(
                                  columns=['categories'],
                                  axis=1).astype('int'),
                              decision_tree_model.predict(validation_set[0])))
                          + ")",
                          "multivariate_analysis/initial/decision_tree")

    # Decision tree plotting
    plotting.graph_exporting(decision_tree_model,
                             training_set[0].columns.tolist(),
                             training_set[1]['y'].unique().astype('str'))

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

    # Decision tree for classification problem
    classification_using_dt(training_set, validation_set)

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
