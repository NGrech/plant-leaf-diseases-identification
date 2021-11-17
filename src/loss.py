"""Module for loss"""
import numpy as np

class CategoricalCrossEntropyLoss:
    """Calculates the CCE loss for a given set of predictions.
    This method expect a softmax output and one-hot encoded label mask
    
    y_pred (np_array): matrix of confidence scores of the prediction
    y_true (np_array): matrix of one-hot encoded true lables of the classes
    """
    def forward(y_pred, y_true):
        # Clipping and applying one hot encoded labels as mask 
        # to zero out scores corresponding to incorrect classes
        # We clip to make sure that none of the reaming classes are 0 or 
        # exactly 1 
        corrected = np.sum(np.clip(y_pred, 1e-7, 1-1e-7)*y_true, axis=1)
        # Taking the -ve log of the remaining confidence scores 
        negative_log = -np.log(corrected)
        return np.mean(negative_log)
    
    def backward(y_pred, y_true):
        """Backpropagation  of the CCE Loss

        Args:
            y_pred (np_array) array of predictions.
            y_true (np_array) array of correct labels.
        """
        return (-y_true/y_pred)/len(y_pred)