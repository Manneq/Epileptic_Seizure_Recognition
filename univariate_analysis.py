import plotting


def univariate_analysis_applying(data):
    plotting.bar_plotting(data, ["Categories", "Patients number"],
                          "Patients EEG categories distribution (numeric)")

    plotting.pie_plotting(data / data.values.sum() * 100,
                          "Patients EEG categories distribution (percentage)")

    return
