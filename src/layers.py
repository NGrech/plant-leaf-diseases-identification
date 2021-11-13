"""Module for Layer classes for neural nets."""
from base import Module
import numpy as np

class LinearLayer(Module):
    """Linear transformation layer of the type o = ixW + b,
    
    where I is the incoming vector, W is the layers weight matrix, b is bias vector and o is the dot product of the 
    i and W plus the bias
    
    Args:
        in_features (int): the size of the input features 
        out_features (int): the size of the output features
        
    Attributes:
        weights (np_array) numpy array of in_features x n_neurons
        biases  (np_array) numpy array of 1 x n_neurons
        inputs  (np_array) numpy array of latest batch of inputs
        inputs  (np_array) numpy array of latest batch of outputs
        d_w     (np_array) The current gradients with respect to the weights 
        d_x     (np_array) The current gradients with respect to the inputs
        d_b     (np_array) The current gradients with respect to the biases
    """

    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        # initializing weights and biases 
        #self.weights = np.random.normal(0.0, np.sqrt(2/in_features), (in_features, out_features))
        # Using a simpler initialization  for testing 
        self.weights = 0.01 * np.random.randn(in_features, out_features)
        self.bias = np.zeros((1, out_features))

    def forward(self, inputs):
        # Saving inputs for backward step
        self.input = inputs
        self.output = np.dot(inputs, self.weights) + self.bias
        return self.output

    def backward(self, d_vals):
        """Backpropagation  of the linear function

        Args:
            d_vals (np_array) array of derivatives from the previous layer/function.
        """
        self.d_w = np.dot(self.input.T, d_vals)
        self.d_x = np.dot(d_vals, self.weights.T)
        self.d_b = np.sum(d_vals, axis=0, keepdims=True)


class Dropout():
    pass

class InputLayer():
    pass

class ConvolutionLayer():
    pass

class PoolingLayer():
    pass