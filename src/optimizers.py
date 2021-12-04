"""Module for optimization functions"""
from layers import LinearLayer
from typing import List

import numpy as np

class SDG:
    """Stochastic Gradient Decent class used to update layer paramers
    The update is the -ve learning rate multiplied by the gradient calculated 
    in the backward step. Optionally it will also apply momentum and decay.

    Args:
        lr       (float): Learning rate to scale the gradients by for the update
        decay    (float): Decay rate used to scale learning rate
        momentum (float): momentum factor used to scale updates to avoid local minima
    
    Attributes:
        lr         (float): Learning rate to scale the gradients by for the update
        clr        (float): Learning rate at the current step
        decay      (float): Decay rate used to scale learning rate
        momentum   (float): momentum factor used to scale updates to avoid local minima
        iterations (int):   Number of times optimizer has completed a step
    """
    IMPLEMENTED = [LinearLayer]

    def __init__(self, learning_rate=1, decay=0., momentum=0.) -> None:
        self.lr = learning_rate
        self.clr = learning_rate # current learning rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0

    def init_momentum(self, layers):
        """Initializes momentum arttribute for layer objects.
        Args:
            Layers (list): A list of layers that need to be updated with momentum.
        """
        for layer in layers:
            if not hasattr(layer, 'momentum_w'):
                layer.momentum_w = np.zeros_like(layer.weights)
                layer.momentum_b = np.zeros_like(layer.bias)

    def pre_update_step(self):
        """Update the current learning rate according to the decay and iterations"""
        decay_rate = 1/(1 + self.decay * self.iterations)
        self.clr = self.lr * decay_rate

    def get_updates(self, layer):
        """Get the update values for a layer's weights and biases
        Args:
            Layers (list): A list of layers that need to be updated with momentum."""
        return (
            -self.clr*layer.d_w,
            -self.clr*layer.d_b
        )

    def get_momentum_updates(self, layer):
        """Updates a layers momentum."""
        wu = (self.momentum * layer.momentum_w) - (self.clr * layer.d_w) 
        bu = (self.momentum * layer.momentum_b) - (self.clr * layer.d_b) 
        layer.momentum_w = wu
        layer.momentum_b = bu
        return (wu, bu)

    def update(self, layers):
        """Update a layers parameters
        Args:
            Layers (list): A list of layers that need to be updated.
        """
        # Test to make sure all layers supported
        if any(l for l in layers if type(l) not in self.IMPLEMENTED):
            unsupported = next(l for l in layers if type(l) not in self.IMPLEMENTED)
            raise NotImplementedError(f'SDG does not support {unsupported.__class__}')

        # pre update step
        if self.decay:
            self.pre_update_step()

        # On the first iteration using momentum initialize the layer momentums
        if self.iterations == 0 and self.momentum:
            self.init_momentum(layers)

        # Update step
        for layer in layers:

            if self.momentum:
                weight_u, bias_u = self.get_momentum_updates(layer)
            else:
                weight_u, bias_u = self.get_updates(layer)
            
            layer.weights += weight_u
            layer.bias += bias_u

        # post update
        self.iterations += 1 

class Adam:
    """Adam Optimizer Short for Adaptive Momentum.
    An extension to the Root mean square propagation (RSMprop) technique that adds in a bias correction mechanism used to correct the momentum and momentum caches.
    To find the update with Adam we need to take the following steps:
        1. Find momentum for the current step
        2. Get corrected the momentum 
        3. Update the cache with the square of the gradient 
        4. Get the corrected cache 
        5. Update weights 
    
    Args:
        learning_rate (float): Learning rate to scale the gradients by for the update
        decay         (float): Decay rate used to scale learning rate
        epsilon       (float): Hyperparmeter for tuning update
        beta_1        (float): Hyperparameter for calculating momentum 
        beta_2        (float): Hyperparameter for calculating cache

    Attributes:
        lr          (float): Learning rate to scale the gradients by for the update
        clr         (float): L:earning rate at current step
        decay       (float): Decay rate used to scale learning rate
        epsilon     (float): Hyperparmeter for tuning update
        beta_1      (float): Hyperparameter for calculating momentum 
        beta_2      (float): Hyperparameter for calculating cache
        iterations  (int):   Number of times optimizer has completed a step
    """

    IMPLEMENTED = [LinearLayer]

    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-8, beta_1=0.9, beta_2=0.999) -> None:
        self.lr = learning_rate
        self.clr = learning_rate # current learning rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iterations = 0

    def pre_update_step(self):
        """Update the current learning rate according to the decay and iterations"""
        decay_rate = 1/(1 + self.decay * self.iterations)
        self.clr = self.lr * decay_rate

    def init_momentum(self, layers):
        """Initializes momentum arttribute for layer objects.
        Args:
            Layers (list): A list of layers that need to be updated with momentum.
        """
        for layer in layers:
            # Init momentum for weights
            layer.momentums_w = np.zeros_like(layer.weights)
            layer.cache_w = np.zeros_like(layer.weights)

            # Init momentums for biases
            layer.momentums_b = np.zeros_like(layer.bias)
            layer.cache_b = np.zeros_like(layer.bias)
            
    def update(self, layers):
        """Update a layers parameters
        Args:
            Layers (list): A list of layers that need to be updated.
        """
        # pre update step
        if self.decay:
           self.pre_update_step()
        
        if self.iterations == 0:
            self.init_momentum(layers)

        # Update step
        for layer in layers:     
            ## Updating momentum 
            layer.momentums_w = self.beta_1 * layer.momentums_w + (1 - self.beta_1) * layer.d_w
            layer.momentums_b = self.beta_1 * layer.momentums_b + (1 - self.beta_1) * layer.d_b

            ## Correcting momentum 
            correction_bias_momentums = 1 - self.beta_1**(self.iterations +1)

            corrected_weights = layer.momentums_w / correction_bias_momentums
            corrected_bias    = layer.momentums_b / correction_bias_momentums

            ## Updating cache
            layer.cache_w = self.beta_2 * layer.cache_w + (1 - self.beta_2) * layer.d_w**2
            layer.cache_b = self.beta_2 * layer.cache_b + (1 - self.beta_2) * layer.d_b**2

            ## Correcting cache
            correction_bias_cache = 1 - self.beta_2**(self.iterations +1)

            corrected_cache_w = layer.cache_w / correction_bias_cache
            corrected_cache_b = layer.cache_b / correction_bias_cache

            ## Updating weights 
            layer.weights += -self.clr * corrected_weights / (np.sqrt(corrected_cache_w) + self.epsilon)

            ## Updating bias
            layer.bias    += -self.clr * corrected_bias / (np.sqrt(corrected_cache_b) + self.epsilon)
        
        # Post update step
        self.iterations += 1
