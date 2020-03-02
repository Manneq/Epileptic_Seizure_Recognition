import pandas as pd


def data_loading():
    data = pd.read_csv("input_data/data.csv", delimiter=',',
                       index_col=None, encoding='utf8')

    data_mapping = pd.read_csv("input_data/categories_mapping.csv",
                               delimiter=',', index_col=None, encoding='utf8')

    data = pd.merge(data, data_mapping, left_on='y', right_on='y', how='inner')
    data = data.set_index('Index')

    return data
