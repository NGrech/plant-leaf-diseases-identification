"""Module for Layer classes for neural nets."""
from base import Module
import numpy as np

class LinearLayer(Module):
    """Linear transformation layer of the type o = ixW + b,
    
    where I is the incoming vector, W is the layers weight matrix, b is bias vector and o is the dot product of the 
    i and W plus the bias
    
    Args:
        in_features      (int):   The size of the input features 
        out_features     (int):   The size of the output features
        lambda_l1_weight (float): Hyperperamiter lambda for L1 regularization for the weights 
        lambda_l1_bias   (float): Hyperperamiter lambda for L1 regularization for the bias
        lambda_l2_weight (float): Hyperperamiter lambda for L2 regularization for the weights
        lambda_l2_bias   (float): Hyperperamiter lambda for L2 regularization for the bias
        
    Attributes:
        weights          (np_array): numpy array of in_features x n_neurons
        biases           (np_array): numpy array of 1 x n_neurons
        inputs           (np_array): numpy array of latest batch of inputs
        outputs          (np_array): numpy array of latest batch of outputs
        d_w              (np_array): The current gradients with respect to the weights 
        d_b              (np_array): The current gradients with respect to the biases
        grad             (np_array): The current gradients with respect to the inputs
        lambda_l1_weight (float):    Hyperperamiter lambda for L1 regularization for the weights 
        lambda_l1_bias   (float):    Hyperperamiter lambda for L1 regularization for the bias
        lambda_l2_weight (float):    Hyperperamiter lambda for L2 regularization for the weights
        lambda_l2_bias   (float):    Hyperperamiter lambda for L2 regularization for the bias
    """

    def __init__(self, in_features, out_features, 
                 lambda_l1_weight=0, lambda_l1_bias=0, 
                 lambda_l2_weight=0, lambda_l2_bias=0) -> None:
        super().__init__()
        # initializing weights and biases 
        self.weights = np.random.normal(0.0, np.sqrt(2/in_features), (in_features, out_features))
        # Using a simpler initialization  for testing 
        #self.weights = 0.01 * np.random.randn(in_features, out_features)
        self.bias = np.zeros((1, out_features))
        # initializing regularization lambdas
        if (lambda_l1_bias > 0) | (lambda_l1_weight > 0):
            self.lambda_l1_weight = lambda_l1_weight
            self.lambda_l1_bias = lambda_l1_bias
        else:
            self.lambda_l1_weight = None
            self.lambda_l1_bias = None
        if (lambda_l2_bias > 0) | (lambda_l2_weight > 0):  
            self.lambda_l2_weight = lambda_l2_weight
            self.lambda_l2_bias = lambda_l2_bias
        else: 
            self.lambda_l2_weight = None
            self.lambda_l2_bias = None

    def forward(self, inputs):
        """Forward pass through the layer.
        
        Args:
        inputs (np_array): Inputs to the layer must be the same size as the weights.
        """
        self.input = inputs
        self.output = np.dot(inputs, self.weights) + self.bias
        return self.output

    def l1_backward_w(self):
        """Backpropagation of L1 regularization function wrt weights."""
        if self.lambda_l1_weight:
            d_l1 = np.ones_like(self.weights) 
            d_l1[self.weights < 0] = -1
            return d_l1 * self.lambda_l1_weight
        else:
            return 0

    def l1_backward_b(self):
        """Backpropagation of L1 regularization function wrt bias."""
        if self.lambda_l1_bias:
            d_l1 = np.ones_like(self.bias) 
            d_l1[self.bias < 0] = -1
            return d_l1 * self.lambda_l1_bias
        else:
            return 0  

    def l2_backward_w(self):
        """Backpropagation of L2 regularization function wrt weights."""
        if self.lambda_l2_weight:
            return 2 * self.lambda_l2_weight  * self.weights
        else:
            return 0
        
    def l2_backward_b(self):
        """Backpropagation of L1 regularization function wrt bias."""
        if self.lambda_l2_bias:
            return 2 * self.lambda_l2_bias  * self.bias
        else:
            return 0

    def backward(self, d_vals):
        """Backpropagation  of the linear layer function

        Args:
            d_vals (np_array): derivatives from the previous layer/function.
        """
        self.d_w = np.dot(self.input.T, d_vals) + self.l1_backward_w() + self.l2_backward_w()
        self.d_b = np.sum(d_vals, axis=0, keepdims=True) + self.l1_backward_b() + self.l2_backward_b()

        self.grad = np.dot(d_vals, self.weights.T)

    def regularization_loss(self):
        """Calculates the regularization loss of the layer. 
        It will only do the calculation if the respective lambda for the loss type is > 0"""
        loss = 0
        # L1 weight 
        if self.lambda_l1_weight:
            loss += self.lambda_l1_weight * np.sum(np.abs(self.weights))
        # L1 bias
        if self.lambda_l1_bias:
            loss += self.lambda_l1_bias * np.sum(np.abs(self.bias))
        # L2 weight
        if self.lambda_l2_weight:
            loss += self.lambda_l2_weight * np.sum(self.weights * self.weights)
        # L2 bias
        if self.lambda_l2_bias:
            loss += self.lambda_l2_bias * np.sum(self.bias * self.bias)

        return loss

class Dropout(Module):
    """ Dropout Layer, intended to be used in traning to deactivate a random portion of the neurons from 
    a pervious layer to based on the work https://arxiv.org/abs/1207.0580

    Args:
        p (float): probability of an element to be set to zero

    Attributes:
        p            (float):   probability of an element to be set to zero
        mask         (ndarray): Latest scaled binary mask used to zero out input elements 
        traning_mode (binary):  Binary flag to control behaviour betwen traning and eval modes
    
    """

    def __init__(self, p:float) -> None:
        super().__init__()
        self.p = p
        self.mask = None
        self.training_mode = True

    def forward(self, input:np.ndarray) -> np.ndarray:
        """During training it will randomly zero out a number of inputs according to a binomial distribution
        and it will also scale the inputs by 1/1-p to account for the lack of dropout in evaluation mode.
        During evaluation it returns the input.
        
        Args:
            input (ndarray): Output from a previous layer
        """
        if not self.training_mode:
            # Eval operation mode -> NO DROPOUT
            self.output = input
            return self.output

        # Training operation mode -> Dropout 
        self.input = input
        self.mask = np.random.binomial(1, self.p, size=input.shape)/ (1-self.p)
        self.output = input * self.mask 
        return self.output
    
    def backward(self, grads):
        """Backpropagation of the dropout function.
        
        Args:
            grads (ndarray): gradients from the next layer.
        """
        self.grad = grads * self.mask
    

class ConvolutionLayer():
    pass

class PoolingLayer():
    pass