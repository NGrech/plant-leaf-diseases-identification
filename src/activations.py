"""Module for activators."""
import numpy as np
from base import Module

class ReLU(Module):
    """Applies Rectified linear Unit function to vector.
    
    Attributes:
        inputs            (ndarray): numpy array of latest batch of inputs
        outputs           (ndarray): numpy array of latest batch of outputs
        grad              (ndarray): The current gradients with respect to the inputs
    """
    def __init__(self) -> None:
        # initializing attributes needed for backwards 
        super().__init__()
        self.grad = None
    
    def forward(self, x):
        # storing inputs needed for backwards 
        self.inputs = x
        self.output = np.maximum(x, 0)
        return self.output
    
    def backward(self, d_vals):
        self.grad = d_vals.copy()
        self.grad[self.inputs <= 0] = 0

class Softmax(Module):
    """Applies Softmax function to input.
    
    Attributes:
        inputs            (ndarray): numpy array of latest batch of inputs
        outputs           (ndarray): numpy array of latest batch of outputs
        grad              (ndarray): The current gradients with respect to the inputs
        confidence_scores (ndarray): Latest batch of classification probabilities
    """

    def __init__(self) -> None:
        super().__init__()
        self.confidence_scores = None

    def forward(self, x):
        """Forward pass
        Args:
            x (ndarray): Input from the pervious layer
        """
        # exponenets of each value
        exp_vals = np.exp(x - np.max(x, axis=1, keepdims=True))
        exp_sum = np.sum(exp_vals, axis=1, keepdims=True)
        # Normalization to get the proabilities 
        self.output = exp_vals/exp_sum
        return self.output

    def _backward(self, d_vals):
        """Backward pass which calculates the gradient wrt the inputs 

        Args:
            d_vals (ndarray): gradients from the loss calculation
        """
        # Initialize array for gradients wrt to inputs
        self.grad = np.zeros_like(d_vals)
        
        _iter = enumerate(zip(self.output, d_vals))
        for i, conf_score, d_val in _iter:
            # Flatten confidence scores
            cs = conf_score.reshape(-1, 1)
            # Find the Jacobian matrix of the output 
            j_matrix = np.diagflat(cs) - np.dot(cs, cs.T)
            # get the gradient 
            self.grad[i] = np.dot(j_matrix, d_val)
    
    def backward(self, y_pred, y_true):
        """Combined backward pass for CCE & Softmax as a single which is
           faster to compute.

        Args:
            y_pred (ndarray): predicted classes for the current batch
            y_true (ndarray): One hot encoded true values for y
        """
        # Number of examples in the batch
        n = len(y_pred)

        # Getting descrete vals from one hot encoding 
        y_true = np.argmax(y_true, axis=1)
        
        self.grad = y_pred.copy()
        self.grad[range(n), y_true] -= 1
        self.grad = self.grad / n
        return self.grad