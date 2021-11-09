"""Module for activators."""
import numpy as np

class ReLU:
    """Applies Rectified linear Unit function to vector."""
    def forward(x):
        return np.maximum(x, 0)

class Softmax:
    """Applies Softmax function to input matrix."""
    def forward(x):
        # exponenets of each value
        exp_vals = np.exp(x - np.max(x, axis=1, keepdims=True))
        exp_sum = np.sum(exp_vals, axis=1, keepdims=True)
        # Normalization to get the proabilities 
        return exp_vals/exp_sum