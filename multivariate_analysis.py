import numpy as np
import data_management
import plotting
import neural_network


def classification_task(data,
                        reduced=False):
    return


def initial_data_analysis(data):
    plotting.heatmap_plotting(data.iloc[:, 0:178].corr(),
                              "Brain activity correlations",
                              "initial", width=3000, height=3000,
                              annotation=False)

    plotting.histogram_plotting(np.std(data, axis=0),
                                ["Parameters", "Standard deviation"],
                                "Initial data standard deviations")

    classification_task(data)

    return


def reduced_data_analysis(data,
                          reduction_type="PCA"):
    data = data_management.dimensionality_reduction(data, reduction_type)

    plotting.data_plotting(data,
                           ["Brain activity X1", "Brain activity X2"],
                           reduction_type + " brain activity", reduction_type)

    plotting.heatmap_plotting(data.iloc[:, 0:2].corr(),
                              reduction_type + " brain activity correlations",
                              reduction_type)

    classification_task(data)

    return


def multivariate_analysis_applying(data):
    initial_data_analysis(data)

    reduced_data_analysis(data)

    reduced_data_analysis(data, reduction_type="FA")

    reduced_data_analysis(data, reduction_type="MDS")

    reduced_data_analysis(data, reduction_type="TSNE")

    return
