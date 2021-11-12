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

    def __init__(self) -> None:
        self.confidence_scores = None

    def forward(self, x):
        # exponenets of each value
        exp_vals = np.exp(x - np.max(x, axis=1, keepdims=True))
        exp_sum = np.sum(exp_vals, axis=1, keepdims=True)
        # Normalization to get the proabilities 
        self.confidence_scores = exp_vals/exp_sum
        return self.confidence_scores

    def backward(self, d_vals):
        # Initialize array for gradients wrt to inputs
        self.d_soft = np.zeros_like(d_vals)
        
        _iter = enumerate(zip(self.confidence_scores, d_vals))
        for i, conf_score, d_val in _iter:
            # Flatten confidence scores
            cs = conf_score.reshape(-1, 1)
            # Find the Jacobian matrix of the output 
            j_matrix = np.diagflat(cs) - np.dot(cs, cs.T)
            # get the gradient 
            self.d_soft[i] = np.dot(j_matrix, d_val)
    
    def combo_backward(self, y_pred, y_true):
        """Does a the combined backward pass for CCE & Softmax as a single, faster step."""
        n = len(y_pred)

        # Getting descrete vals from one hot encoding 
        y_true = np.argmax(y_true, axis=1)
        
        self.d_soft = y_pred.copy()
        self.d_soft[range(n), y_true] -= 1
        self.d_soft = self.d_soft/n
        return self.d_soft

