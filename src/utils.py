"""Modules for common utility functions."""
import numpy as np

def one_hot_encode_index(y, n):
    return np.eye(n)[y]

def accuracy(y_pred, y_true):
    """Calculates the accuracy of a batch of predictions"""
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))