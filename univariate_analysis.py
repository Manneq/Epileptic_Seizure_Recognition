"""
    File 'univariate_analysis.py' used to find distributions for categories.
"""
import plotting


def univariate_analysis_applying(data):
    """
        Method to perform univariate analysis on categories.
        param:
            data - pandas DataFrame of mapped data
    """
    # Categories distribution plotting
    plotting.bar_plotting(data, ["Categories", "Patients number"],
                          "Patients EEG categories distribution (numeric)",
                          "univariate_analysis")

    # Categories distribution plotting (percents)
    plotting.pie_plotting(data / data.values.sum() * 100,
                          "Patients EEG categories distribution (percentage)",
                          "univariate_analysis")

    return
