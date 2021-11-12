"""Module for Layer classes for neural nets."""
import numpy as np

class Layer:
    """Base class"""
    def __init__(self) -> None:
        pass

    def forward(self):
        pass
    
    def backward(self):
        pass

class LinearLayer():
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
        d_w     (np_array) The current gradients with respect to the weights 
        d_x     (np_array) The current gradients with respect to the inputs
        d_b     (np_array) The current gradients with respect to the biases
    """

    def __init__(self, in_features, out_features) -> None:
        # initializing weights and biases 
        self.weights = np.random.normal(0.0, np.sqrt(2/in_features), (in_features, out_features))
        self.bias = np.zeros((1, out_features))
        # initializing attributes needed for backwards 
        self.inputs = None

    def forward(self, inputs):
        # Saving inputs for backward step
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.bias

    def backward(self, d_vals):
        """Backpropagation  of the linear function

        Args:
            d_vals (np_array) array of derivatives from the previous layer/function.
        """
        self.d_w = np.dot(self.inputs.T, d_vals)
        self.d_x = np.dot(d_vals, self.weights.T)
        self.d_b = np.sum(d_vals, axis=0, keepdims=True)

class Dropout(Layer):
    pass

class InputLayer(Layer):
    pass

class ConvolutionLayer(Layer):
    pass

class PoolingLayer(Layer):
    pass