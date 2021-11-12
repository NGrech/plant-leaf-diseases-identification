"""Modules for common utility functions."""
import numpy as np

def one_hot_encode_index(y, n):
    return np.eye(n)[y]