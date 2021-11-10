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

class LinearLayer(Layer):
    """Linear transformation layer of the type o = ixW + b,
    
    where I is the incoming vector, W is the layers weight matrix, b is bias vector and o is the dot product of the 
    i and W plus the bias
    
    Args:
        in_features (int): the size of the input features 
        out_features (int): the size of the output features
        
    Attributes:
        weights (np_array) numpy array of in_features x n_neurons
        biases (np_array) numpy array of 1 x n_neurons
    """

    def __init__(self, in_features, out_features) -> None:
        # initializing weights and biases 
        self.weights = np.random.normal(0.0, np.sqrt(2/in_features), (in_features, out_features))
        self.bias = np.zeros((1, out_features))

    def forward(self, inputs):
        # Storing output for backpropagation 
        self.output = np.dot(inputs, self.weights) + self.bias
        return self.output


class Dropout(Layer):
    pass

class InputLayer(Layer):
    pass

class ConvolutionLayer(Layer):
    pass

class PoolingLayer(Layer):
    pass