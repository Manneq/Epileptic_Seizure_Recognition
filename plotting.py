import numpy as np
import matplotlib.pyplot as plt


def bar_plotting(data, labels, title, folder,
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
    plt.savefig("plots/" + folder + "/" + title + ".png", dpi=dpi)
    plt.close()

    return


def pie_plotting(data, title, folder,
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
    plt.savefig("plots/" + folder + "/" + title + ".png", dpi=dpi)
    plt.close()

    return
