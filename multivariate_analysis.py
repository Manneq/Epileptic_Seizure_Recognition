"""
    File 'multivariate_analysis.py' used for multivariate analysis.
"""
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

    # Training and validation sets creation
    training_set, validation_set = data_management.sets_creation(data)

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
