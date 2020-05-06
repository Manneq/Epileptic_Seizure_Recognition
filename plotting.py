import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def bar_plotting(data, labels, title,
                 width=1920, height=1080, dpi=96, font_size=22, color='b'):
    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.rcParams.update({'font.size': font_size})
    plt.bar((np.arange(len(data))), data, align='center',
            color=color)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.xticks(np.arange(len(data)), data.index)
    plt.title(title)
    plt.tight_layout()
    plt.savefig("output_data/univariate_analysis/" + title + ".png", dpi=dpi)
    plt.close()

    return


def pie_plotting(data, title,
                 width=1920, height=1080, dpi=96, font_size=22):
    explode = ()

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
    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.rcParams.update({'font.size': font_size})
    sns.heatmap(data, annot=annotation).set_title(title)
    plt.savefig("output_data/multivariate_analysis/" + folder + "/" +
                title + ".png", dpi=dpi)
    plt.close()

    return


def data_plotting(data, labels, title, folder,
                  width=1920, height=1080, dpi=96, font_size=22):
    color_palette = ['r', 'g', 'b', 'c', 'm']

    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.rcParams.update({'font.size': font_size})

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


def histogram_plotting(data, labels, title,
                       width=1920, height=1080, dpi=96, font_size=22):
    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.rcParams.update({'font.size': font_size})
    plt.hist(data, bins=data.shape[0])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig("output_data/multivariate_analysis/initial/" +
                title + ".png", dpi=dpi)
    plt.close()

    return
