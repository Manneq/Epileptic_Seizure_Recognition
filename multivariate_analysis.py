import pandas as pd
import sklearn.decomposition
import sklearn.manifold
import numpy as np
import plotting


def dimensionality_reduction_and_analysis(data, reduction_type):
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

    data_transformed = \
        pd.DataFrame(np.append(reduction_model.fit_transform(
            data.iloc[:, 0:178]), data.iloc[:, 178:180].values, axis=1),
                     columns=['x1', 'x2', 'y', 'categories'])

    plotting.data_plotting(data_transformed,
                           ["Brain activity X1", "Brain activity X2"],
                           reduction_type + " brain activity",
                           "multivariate_analysis/" + reduction_type)

    plotting.heatmap_plotting(data.iloc[:, 0:2].corr(),
                              reduction_type + " brain activity correlations",
                              "multivariate_analysis/" + reduction_type)

    return


def multivariate_analysis_applying(data):
    plotting.heatmap_plotting(data.iloc[:, 0:178].corr(),
                              "Brain activity correlations",
                              "multivariate_analysis",
                              width=3000, height=3000, annotation=False)

    dimensionality_reduction_and_analysis(data, reduction_type="PCA")

    dimensionality_reduction_and_analysis(data, reduction_type="FA")

    dimensionality_reduction_and_analysis(data, reduction_type="MDS")

    dimensionality_reduction_and_analysis(data, reduction_type="TSNE")

    return
