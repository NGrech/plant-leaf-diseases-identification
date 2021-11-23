"""Module for Layer classes for neural nets."""
from base import Module
import numpy as np
from scipy.signal import convolve2d
import itertools as it

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
    
class Layer2d(Module):
    
    """
    Base-class for 2d-layers
    
    Args:
        channels (int):     number of input channels
        kernel_size (int):  size of convolution kernel
        
    Attributes:
        kernels (int):      number of kernels in kernel 
        kernel (np_array):  numpy array of a number of 2d kernels
        bias (np_array):    numpy array of the bias
        inputs (np_array):  numpy array of latest batch of inputs with the dims (batch_size, channels, height, width)
        outputs (np_array): numpy array of latest batch of outputs
        batch_size (int):   number of images in latest inputs

        r_im (range):       range for batch    
        r_ch (range):       range for channels

        dinputs (np_array):     The current gradients with respect to the layer input
        dkernel (np_array):     The current gradients with respect to the kernel weights
        dbiases (np_array):     The current gradients with respect to the biases
    """

    def __init__(self, channels, kernel_size):

        self.channels = channels
        self.kernel_size = kernel_size

        self.r_ch = range(self.channels)

        self.kernels = 1

        # Filter kernel definition - Low-pass
        self.kernel = np.ones([self.kernels, kernel_size, kernel_size])

        self.pad = 0
        self.stride = 1

    def forward(self, inputs):
        
        """Forward pass through a 2d-layer

        Args:
            inputs (np_array): numpy array of latest batch of inputs with the dims (batch_size, channels, height, width).
        """

        self.inputs = inputs
        self.batch_size = self.inputs.shape[0]
        self.bias = np.zeros([self.batch_size, self.channels])

        self.r_im = range(self.batch_size)

    def backward(self, dvalues):

        """Backward pass through a 2d-layer

        Args:
            dvalues (np_array): array of derivatives from the previous layer/function.
        """

        self.dinputs = np.zeros_like(self.inputs)

class ConvolutionLayer(Layer2d):
    
    """
    Convolutional transformation layer
    
    Args:
        kernel_size (int):  size of convolution kernel
        
    Attributes:
        kernels (int):      number of kernel in kernel 
        kernel (np_array):  numpy array of a number of 2d kernels

        dkernel (np_array):     The current gradients with respect to the kernel weights
        dbiases (np_array):     The current gradients with respect to the biases
    """

    def __init__(self, channels, kernel_size):

        super().__init__(channels, kernel_size)

    def forward(self, inputs):
        
        """Forward pass through the convolution layer

        Args:
            inputs (np_array): numpy array of latest batch of inputs with the dims (batch_size, channels, height, width).
        """

        super().forward(inputs)

        self.outputs = self.__convolve2d()

    def backward(self, dvalues):

        """Backpropagation through the convolution layer

        Args:
            dvalues (np_array): array of derivatives from the previous layer/function.
        """

        super().backward(dvalues)

        self.dkernel = np.zeros_like(self.kernel)

         # Calculate the loss-gradient with repect to the kernel weights (dL/dk = dL/dO * dO/dX, dL/dO = dvalues)
         # => dL/dk = convolution(local inputs, back-propagated derivatives)
        for im, ch in it.product(self.r_im, self.r_ch):
            c1 = self.inputs[im, ch, :, :]
            c2 = dvalues[im, ch, :, :]
            self.dkernel[0, :, :] = convolve2d(c1, c2, mode='valid')

        # Calculate the loss-gradient with repect to the layer inputs (dL/dX = dL/dO * dO/dk, dL/dO = dvalues)
        #  => dL/dX = full-convolution(back-propagated derivatives, local 180-deg rotate filter)
        for im, ch in it.product(self.r_im, self.r_ch):
            c1 = dvalues[im, ch, :, :]
            c2 = np.rot90(self.kernel[0, :, :], 2)
            self.dinputs[im, ch, :, :] = convolve2d(c1, c2, mode='full')

        # Calculate the loss-gradient with respect to each bias. It sums over the batch/sample-, image-height-
        # and image-width-dimensions.
        self.dbiases = np.sum(dvalues, axis=(0, 2, 3), keepdims=True)

    def __convolve2d(self, pad=0, strides=1):

        ER = self.inputs.shape[2]
        EC = self.inputs.shape[3]

        k_ER = self.kernel.shape[1]
        k_EC = self.kernel.shape[2]

        out_ER = int(((ER - k_ER + 2 * pad) / strides) + 1)
        out_EC = int(((EC - k_EC + 2 * pad) / strides) + 1)
        output = np.zeros([self.inputs.shape[0], self.channels, out_ER, out_EC])
        # output = np.zeros([self.inputs.shape[0], self.channels, ER, EC])

        for im, ch in it.product(self.r_im, self.r_ch):
            t1 = self.inputs[im, ch, :, :]
            t2 = self.kernel[0, :, :]
            output[im, ch, :, :] = convolve2d(t1, t2, mode='valid') + self.bias[im, ch]

        return output

class PoolingLayer(Layer2d):

    """
    Pooling transformation layer
    
    Args:
        channels (int):     number of input channels
        kernel_size (int):  size of the kernel
        
    Attributes:
        kernels (int):      number of kernels in kernel
        kernel (np_array):  numpy array of a number of 2d kernels
    """

    def __init__(self, channels, kernel_size):

        super().__init__(channels, kernel_size)


    def forward(self, inputs):
        
        """Forward pass through the pooling layer

        Args:
            inputs (np_array): array of derivatives from the previous layer/function.
        """

        super().forward(inputs)

        self.outputs = self.__pool2d(0, 1)

    def backward(self, dvalues):

        """Backward pass through a max-pooling-layer

        Args:
            dvalues (np_array): array of derivatives from the previous layer/function.
        """

        super().backward(dvalues)

        self.dinputs = self.__derivative(dvalues)


    def __derivative(self, dvalues):

        k_ER = self.kernel.shape[1]
        k_EC = self.kernel.shape[2]

        ER = self.inputs.shape[2]
        EC = self.inputs.shape[3]

        r_ER_i = range(0, ER, k_ER)
        r_EC_i = range(0, EC, k_EC)
        # r_ER_i = range(int(k_ER/2), ER - int(k_ER/2), self.kernel_size)
        # r_EC_i = range(int(k_EC/2), EC - int(k_EC/2), self.kernel_size)

        output = np.zeros_like(self.inputs)

        for im, ch in it.product(self.r_im, self.r_ch):
        # for im, ch in it.product(range(1), range(1)):

            o_R = 0
            for R in r_ER_i:        
                o_C = 0
                for C in r_EC_i:
                    
                    # Get each kernel-sampled value from inputs
                    val = self.inputs[im, ch, R: R + r_ER_i.step, C: C + r_EC_i.step]
                    # val = self.inputs[im, ch, R - int(k_ER/2): R + int(k_ER/2), C - int(k_EC/2): C + int(k_EC/2)]

                    # Create a mask based on max value(s) in each kernel
                    mask = (val == val.max()).astype(int)

                    print(mask)

                    # Calculate gradient with respect to the inputs
                    output[im, ch, R: R + r_ER_i.step, C: C + r_EC_i.step] = mask * dvalues[im, ch, o_R, o_C]

                    o_C += 1
                o_R += 1

        return output
        

    def __pool2d(self, pad, strides):

        k_ER = self.kernel.shape[1]
        k_EC = self.kernel.shape[2]

        ER = self.inputs.shape[2]
        EC = self.inputs.shape[3]

        r_ER_i = range(0, ER, k_ER)
        r_EC_i = range(0, EC, k_EC)
        # r_ER_i = range(int(k_ER/2), ER - int(k_ER/2), self.kernel_size)
        # r_EC_i = range(int(k_EC/2), EC - int(k_EC/2), self.kernel_size)

        output = np.zeros([self.batch_size, self.channels, len(r_ER_i) + pad*2, len(r_EC_i) + pad*2])

        for im, ch in it.product(self.r_im, self.r_ch):
        # for im, ch in it.product(range(1), range(1)):
            o_R = 0
            for R in r_ER_i:        
                o_C = 0
                for C in r_EC_i:
                    val = self.inputs[im, ch, R: R + r_ER_i.step, C: C + r_EC_i.step].max()
                    # val = self.inputs[im, ch, R - int(k_ER/2): R + int(k_ER/2), C - int(k_EC/2): C + int(k_EC/2)].max()
                    output[im, ch, pad + o_R, pad + o_C] = val
                    o_C += 1
                o_R += 1

        return output