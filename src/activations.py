"""Module for activators."""
import numpy as np

class ReLU:
    """Applies Rectified linear Unit function to vector."""
    def __init__(self) -> None:
        # initializing attributes needed for backwards 
        self.inputs = None
        self.d_relu = None
    
    def forward(self, x):
        # storing inputs needed for backwards 
        self.inputs = x
        return np.maximum(x, 0)
    
    def backward(self, d_vals):
        self.d_relu = d_vals.copy()
        self.d_relu[self.inputs <= 0] = 0

class Softmax:
    """Applies Softmax function to input matrix."""
    def forward(x):
        # exponenets of each value
        exp_vals = np.exp(x - np.max(x, axis=1, keepdims=True))
        exp_sum = np.sum(exp_vals, axis=1, keepdims=True)
        # Normalization to get the proabilities 
        return exp_vals/exp_sum