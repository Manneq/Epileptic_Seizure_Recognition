import data_management
import univariate_analysis
import multivariate_analysis


def main():
    data = data_management.data_loading()

    """univariate_analysis.univariate_analysis_applying(
        data[['categories']].groupby('categories').size())"""

    multivariate_analysis.multivariate_analysis_applying(data)

    return


if __name__ == "__main__":
    main()
