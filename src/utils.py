"""Modules for common utility functions."""
import numpy as np

def one_hot_encode_index(y, n):
    one_hot = np.zeros((len(y), n))
    for _y, oh in zip(y, one_hot):
        oh[_y] = 1
    return one_hot