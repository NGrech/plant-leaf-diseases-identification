"""Module for activators."""
import numpy as np

class ReLU:
    """Applies Rectified linear Unit function to vector."""
    def forward(x):
        return np.maximum(x, 0)

class Softmax():
    pass