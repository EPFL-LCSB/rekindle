import numpy as np
import pickle
import pandas as pd


def read_with_pd(path, delimiter='\t', header=None):
    data_pd = pd.read_csv(path, delimiter=delimiter, header=header)
    return data_pd[0].tolist()


def save_pkl(name, obj):
    """save obj with pickle"""
    name = name.replace('.pkl', '')
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pkl(name):
    """load a pickle object"""
    name = name.replace('.pkl', '')
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def write_in_file(path_to_file, data):
    with open(path_to_file, 'w+') as f:
        for item in data:
            f.write("%s\n" % item)


def scale_range(x, a, b):
    """scale an input between a and b, b>a"""
    assert b > a

    min_x = np.min(x)
    max_x = np.max(x)

    x_scaled = (b - a) * np.divide(x - min_x, max_x - min_x) + a

    return x_scaled, min_x, max_x


def unscale_range(x_scaled, a, b, min_x, max_x):
    """unscale a scaled input"""
    assert b > a
    assert max_x > min_x

    x = np.divide(x_scaled - a, b - a) * (max_x - min_x) + min_x

    new_min = np.min(x)
    new_max = np.max(x)

    return x, new_min, new_max

