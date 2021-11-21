"""Module for loss"""
import numpy as np
from numpy.core.numeric import array_equal

class CategoricalCrossEntropyLoss:
    """Calculates the CCE loss for a given set of predictions.
    This method expect a softmax output and one-hot encoded label mask
    
    Attributes:
        acc_sum     (ndarray): accumulated sum of negative log loss
        acc_count   (int):     accumulated count of number of samples seen
    """
    def __init__(self) -> None:
        self.acc_sum = 0
        self.acc_count = 0


    def forward(self, y_pred, y_true):
        """Calculate the mean negative log loss and accumulate it 

        Args:
            y_pred (np_array): matrix of confidence scores of the prediction
            y_true (np_array): matrix of one-hot encoded true lables of the classes
        """
        # Clipping and applying one hot encoded labels as mask 
        # to zero out scores corresponding to incorrect classes
        # We clip to make sure that none of the reaming classes are 0 or 
        # exactly 1 
        clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        corrected = np.sum(clipped*y_true, axis=1)
        # Taking the -ve log of the remaining confidence scores 
        negative_log = -np.log(corrected)

        # accumulating loss
        self.acc_sum += np.sum(negative_log)
        self.acc_count += len(negative_log)

        return np.mean(negative_log)

    def get_accumulated_loss(self):
        return self.acc_sum / self.acc_count

    def reset(self):
        self.acc_count = 0
        self.acc_sum = 0

    def backward(y_pred, y_true):
        """Backpropagation  of the CCE Loss

        Args:
            y_pred (np_array) array of predictions.
            y_true (np_array) array of correct labels.
        """
        return (-y_true/y_pred)/len(y_pred)

class Accuracy:
    """
    Class to calculate the accuracy.

    Attributes:
        current_accuracy    (float):   Latest accuracy recorded
        sum                 (ndarray): Accumulated sum of accuracy comparisons 
        count               (int):     Accumulated count of number of samples
    """

    def __init__(self) -> None:
        self.current_accuracy = 0
        self.sum = 0
        self.count = 0

    def forward(self, y_pred, y_true):
        """Calculates the accuracy of a batch of predictions"""
        comp = np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)
        self.sum += np.sum(comp)
        self.count += len(comp)
        return np.mean(comp)

    def get_accumulated_accuracy(self):
        return self.sum/self.count

    def reset(self):
        self.count = 0
        self.sum = 0