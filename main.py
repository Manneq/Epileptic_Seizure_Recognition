import data_management
import univariate_analysis
import multivariate_analysis
import modeling


"""
    File 'main.py' is main file that controls the sequence of function calls.
"""


def main():
    """
        Main function.
    """
    # Data import
    data = data_management.data_loading()

    # Univariate analysis
    univariate_analysis.univariate_analysis_applying(
        data['categories'].groupby('categories').size())

    # Multivariate analysis
    multivariate_analysis.multivariate_analysis_applying(data)

    # Modeling and forecasting
    modeling.modeling_applying(data)

    return


if __name__ == "__main__":
    main()
