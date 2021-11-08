"""Module for Layer classes for neural nets."""
import numpy as np

class FullyConnectedLayer():
    """Fully connected or dense layer of the network."""

    def __init__(self, inputs, depth) -> None:
        """
        Any fully connected layer in the networks
        
        Args:
            inputs (int) : number of inputs the layer should expect
        """
        self.n_neurons = depth
        self.weights = np.random.randn(input, depth)
        self.bias = np.zeros((1, depth))

    def forward(self, inputs):
        """
        Forward pass of network

        Args:
            inputs (np.array) vector of input values 
        """

        self.output = np.dot(inputs, self.weights) + self.bias

        return self.output



class Dropout():
    pass

class InputLayer():
    pass

class ConvolutionLayer():
    pass

class PoolingLayer():
    pass