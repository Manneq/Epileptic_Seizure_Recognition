import data_import
import plotting


def univariate_analysis_applying(data):
    plotting.bar_plotting(data, ["Categories", "Patients number"],
                          "Patients EEG categories distribution (numeric)",
                          "univariate_analysis")

    data = data / data.values.sum() * 100

    plotting.pie_plotting(data,
                          "Patients EEG categories distribution (percentage)",
                          "univariate_analysis")

    return


def main():
    data = data_import.data_loading()

    univariate_analysis_applying(
        data[['categories']].groupby('categories').size())

    return


if __name__ == "__main__":
    main()
