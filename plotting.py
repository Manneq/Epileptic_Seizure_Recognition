"""
    File 'plotting.py' has functions for plotting different data.
"""
import pandas as pd
import sklearn.tree
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def bar_plotting(data, labels, title, folder,
                 width=1920, height=1080, dpi=96, font_size=22, color='b'):
    """
        Method to plot bar chart.
        param:
            1. data - pandas Series of data that should be plotted
            2. labels - vector of strings (2) that are
                x label and y label of plot
            3. title - string name of plot
            4. folder - string folder path
            5. width - int value of plot width in pixels (1920 as default)
            6. height - int value of plot height in pixels (1080 as default)
            7. dpi - int value of plot dpi (96 as default)
            8. font_size - int value of text size on plot (22 as default)
            9. color - string value of color name for plot ('b' as default)
    """
    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.rcParams.update({'font.size': font_size})
    plt.bar((np.arange(len(data))), data, align='center',
            color=color)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.xticks(np.arange(len(data)), data.index)
    plt.title(title)
    plt.tight_layout()
    plt.savefig("output_data/" + folder + "/" + title + ".png", dpi=dpi)
    plt.close()

    return


def pie_plotting(data, title,
                 width=1920, height=1080, dpi=96, font_size=22):
    """
        Method to plot pie chart.
        param:
            1. data - pandas DataFrame of data that should be plotted
            2. title - string name of plot
            3. width - int value of plot width in pixels (1920 as default)
            4. height - int value of plot height in pixels (1080 as default)
            5. dpi - int value of plot dpi (96 as default)
            6. font_size - int value of text size on plot (22 as default)
    """
    explode = ()

    # Explode for pie chart pieces
    for i in range(len(data)):
        explode += (0.1, )

    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.rcParams.update({'font.size': font_size})
    plt.pie(data, explode=explode, labels=data.index,
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')
    plt.title(title)
    plt.tight_layout()
    plt.savefig("output_data/univariate_analysis/" + title + ".png", dpi=dpi)
    plt.close()

    return


def heatmap_plotting(data, title, folder,
                     width=1920, height=1080, dpi=96, font_size=22,
                     annotation=True):
    """
        Method to plot correlations between parameters.
        param:
            1. data - numpy array of correlations that should be plotted
            2. title - string name of plot
            3. folder - string name of folder
            4. width - int value of plot width in pixels (1920 as default)
            5. height - int value of plot height in pixels (1080 as default)
            6. dpi - int value of plot dpi (96 as default)
            7. font_size - int value of text size on plot (22 as default)
            8. annotation - boolean value for using annotation
                (True as default)
    """
    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.rcParams.update({'font.size': font_size})
    sns.heatmap(data, annot=annotation).set_title(title)
    plt.savefig("output_data/multivariate_analysis/" + folder + "/" +
                title + ".png", dpi=dpi)
    plt.close()

    return


def data_plotting(data, labels, title, folder,
                  width=1920, height=1080, dpi=96, font_size=22):
    """
        Method to plot reduced data.
        param:
            1. data - pandas DataFrame of data that should be plotted
            2. labels - vector of strings (2) that are
                x label and y label of plot
            3. title - string name of plot
            4. folder - string name of folder
            5. width - int value of plot width in pixels (1920 as default)
            6. height - int value of plot height in pixels (1080 as default)
            7. dpi - int value of plot dpi (96 as default)
            8. font_size - int value of text size on plot (22 as default)
    """
    # Color palette for points
    color_palette = ['r', 'g', 'b', 'c', 'm']

    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.rcParams.update({'font.size': font_size})

    # Points plotting with legend
    for category_number in np.unique(data['y'].values):
        data_selected = data[data['y'] == category_number]

        plt.scatter(data_selected.iloc[:, 0], data_selected.iloc[:, 1],
                    c=color_palette[category_number],
                    label=data_selected.iloc[0, 3])

    plt.legend()
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig("output_data/multivariate_analysis/" +
                folder + "/" + title + ".png", dpi=dpi)
    plt.close()

    return


def histogram_plotting(data, labels, title, folder,
                       width=1920, height=1080, dpi=96, font_size=22):
    """
        Method to plot histogram.
        param:
            1. data - numpy array of data that should be plotted
            2. labels - vector of strings (2) that are
                x label and y label of plot
            3. title - string name of plot
            4. folder - string name of folder
            5. width - int value of plot width in pixels (1920 as default)
            6. height - int value of plot height in pixels (1080 as default)
            7. dpi - int value of plot dpi (96 as default)
            8. font_size - int value of text size on plot (22 as default)
    """
    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.rcParams.update({'font.size': font_size})
    plt.hist(data, bins=int(np.sqrt(data.shape[0])))
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig("output_data/" + folder + "/" + title + ".png", dpi=dpi)
    plt.close()

    return


def graph_exporting(data, feature_names, class_names, folder):
    """
        Method to export decision tree example.
        param:
            1. data - fitted sklearn random forest
            2. feature_names - list of names of features
            3. class_names - list of class names
            4. folder - string name of folder
    """
    sklearn.tree.export_graphviz(data.estimators_[0],
                                 out_file="output_data/multivariate_analysis/"
                                          "initial/random_forest/" +
                                          folder + "/tree_example.dot",
                                 feature_names=feature_names,
                                 class_names=class_names,
                                 rounded=True, proportion=False,
                                 precision=2, filled=True, label='all')

    # To png exporting
    os.system("dot -Tpng output_data/multivariate_analysis/initial/"
              "random_forest/" + folder + "/tree_example.dot -o output_data/"
                                          "multivariate_analysis/initial/"
                                          "random_forest/" + folder +
              "/tree_example.png")

    # dot file deleting
    os.remove("output_data/multivariate_analysis/initial/random_forest/" +
              folder + "/tree_example.dot")

    return


def line_plotting(data, labels, title, folder,
                  width=1920, height=1080, dpi=96, font_size=22,
                  forecasted_data=None, distribution=False):
    """
        Method to plot line chart of forecasting and distributions.
        param:
            1. data - pandas DataFrame of data
            2. labels - tuple of string labels
            3. title - string name of plot
            4. width - int value of plot width in pixels (1920 as default)
            5. height - int value of plot height in pixels (1080 as default)
            6. dpi - int value of plot dpi (96 as default)
            7. font_size - int value of text size on plot (22 as default)
            8. forcasted_data - pandas DataFrame of forecasted data
                (None as default)
            9. distribution - boolean value for chart type selection
                (False as default)
    """
    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.rcParams.update({'font.size': font_size})

    if distribution:
        # Mean distribution plotting
        plt.plot(data.index, data.values, 'b-')
        plt.xticks(np.arange(len(data)), data.index, rotation=90)
    else:
        # Forecasting plotting
        plt.plot(data.index, data.values, 'b-', label="Brain activity")

        if isinstance(forecasted_data, pd.DataFrame):
            plt.plot(forecasted_data.index.tolist(), forecasted_data.values,
                     'r-', label="Forecasted brain activity")

        plt.legend()

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig("output_data/" + folder + "/" + title + ".png", dpi=dpi)
    plt.close()

    return
