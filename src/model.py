"""Module for model creating, loading and saving"""
import mlflow
import os
import copy
import pickle
import numpy as np
from layers import Dropout
from loss import Accuracy
from time import time
from datetime import timedelta


class Model:
    """Model class designed as a container to simplify the building and training of networks.
    
    Args:
    optimizer:      The optimizer that should be used
    Loss:           The loss class that should be used
    experiment_id:  MLflow experiment Id 

    Attributes:
        layers              (list):    List of all layers in the network in their activation sequence 
        trainable_layers    (list):    List of trainable layers in the network
        loss                ():        The loss class that should be used
        optim               ():        The optimizer that should be used
        current_loss        (float):   Latest loss recorded 
        training_mode       (bool):    Boolean flag if the network is in training mode
    """

    def __init__(self, optimizer, loss) -> None:
        self.layers = []
        self.trainable_layers = []
        self.loss = loss
        self.accuracy = Accuracy()
        self.optim = optimizer
        self.current_loss = 0
        self.training_mode = True
        self.val_loss = None
        self.start_time = None
        self.acc_time = 0

    def set_save_config(self, model_name=None, save_path=None):
        """Configure path and name of model."""
        self.model_name = ('my_model', model_name)[model_name!=None]
        _base_pth = ('.', save_path)[save_path!=None]
        self.save_path = os.path.join(_base_pth, self.model_name)

    def add_early_stop(self, treshold):
        """Add early stop functionality to model"""
        self.early_stop_treshold = treshold
        self.early_stop_counter = 0
        self.early_stop_prev_val = np.Inf

    def check_early_stop(self):
        """Checking for early stop condition"""
        if self.loss.get_accumulated_loss() < self.early_stop_prev_val:
            self.early_stop_prev_val = self.loss.get_accumulated_loss()
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1

        if self.early_stop_counter >= self.early_stop_treshold:
            return True

        return False

    def __repr__(self) -> str:
        """Custom dunder representer method to print out all the layers of the network."""
        layer_str = "".join([f"\t ({i}): {type(l).__name__} (Trainable: {l in self.trainable_layers})\n" 
                            for i, l in enumerate(self.layers)])

        return "Model Architecture: \n" + layer_str
    
    def add(self, layer):
        """Appends a single layer to the end of the network"""
        self.layers.append(layer)
        if hasattr(layer, 'weights'):
            self.trainable_layers.append(layer)

    def set_sequence(self, layers):
        """Defines an entire sequence of layers. NOTE: will overwrite current network"""
        self.layers = layers
        self.trainable_layers = [l for l in layers if hasattr(l, 'weights')]

    def get_loss(self, y_pred, y_true):
        """Calculates the current loss of the network"""
        # Calculating loss and regularized loss  
        loss = self.loss.forward(y_pred, y_true)
        # Regularised loss
        rl = sum([l.regularization_loss() for l in self.trainable_layers if hasattr(l, 'regularization_loss')])
        return loss + rl

    def forward(self, X):
        """Handles forward pass through all layers"""
        # First layer
        self.layers[0].forward(X)

        # Rest of layers 
        for i in range(1, len(self.layers)):
            self.layers[i].forward(self.layers[i-1].output)

    def backward(self, y_pred, y_true):
        """Handles backward pass through all layers"""
        # Last layer
        self.layers[-1].backward(y_pred, y_true)

        # Rest of the layers
        for i, l in reversed(list(enumerate(self.layers[:-1]))):
            l.backward(self.layers[i+1].grad)
        
    def step_logger(self, step, steps):
        """Log current state of model"""
        print(
            f"Step: {step}/{steps}, accuracy{self.current_accuracy:.3f}, loss{self.current_loss:.3f}, learning rate {self.optim.clr:.7f} "
        )

    def epoch_logger(self, epoch, epochs):
        """log current state of model"""
        epoch_time = time() - self.start_time
        self.acc_time += epoch_time
        if mlflow.active_run():
            mlflow.log_metric('accuracy', self.current_accuracy, step=epoch)
            mlflow.log_metric('loss', self.current_loss, step=epoch)
            mlflow.log_metric('learning rate', self.optim.clr, step=epoch)
            mlflow.log_metric('execution time', epoch_time, step=epoch)

        print(
            f"Epoch: {epoch+1}/{epochs}, accuracy{self.accuracy.get_accumulated_accuracy():.3f}, loss{self.loss.get_accumulated_loss():.3f}, learning rate {self.optim.clr:.3f}"
        )
        pred_time = (epochs*self.acc_time)/(epoch+1)
        print(f"Estimated reamining runtime: {str(timedelta(seconds=pred_time))}")
    
    def validate(self, X_val, y_val, batch_size, steps, epoch=None, save_model=True):
        """Handles the validation pass of the network"""
        self.mode_eval()
        self.reset()

        for step in range(steps):
            X_batch, y_batch = self.batch_data(step, batch_size, X_val, y_val)

            self.forward(X_batch)

            self.loss.forward(self.layers[-1].output, y_batch)
            self.accuracy.forward(self.layers[-1].output, y_batch)

        current_vall_loss = self.loss.get_accumulated_loss()

        if hasattr(self, 'save_path'):
            if self.val_loss:
                if self.val_loss > current_vall_loss and save_model:
                    self.val_loss = current_vall_loss
                    print('New best model ... saving')
                    self.save(self.save_path)
            else:
                # Not saving first version of model 
                self.val_loss = current_vall_loss

        if mlflow.active_run():
            mlflow.log_metric('validation_accuracy', self.accuracy.get_accumulated_accuracy(), step=epoch)
            mlflow.log_metric('validation_loss', current_vall_loss, step=epoch)

        print(
            f"Validation : Accuracy: {self.accuracy.get_accumulated_accuracy():.3f}, Loss: {self.loss.get_accumulated_loss():.3f}"
        )
    
    def evaluate(self, X_val, y_val, batch_size):
        val_steps = self.get_steps(X_val, batch_size)
        self.validate(X_val, y_val, batch_size, val_steps, save_model=False)

    def mode_train(self):
        """Sets the model and all dropout layers to training mode."""
        if not self.training_mode:
            for l in self.layers:
                if type(l) == Dropout:
                    l.training_mode = True
        self.training_mode = True
    
    def mode_eval(self):
        """Sets the model and all dropout layers to evaluation mode."""
        if self.training_mode:
            for l in self.layers:
                if type(l) == Dropout:
                    l.training_mode = False
        self.training_mode = False
        
    def get_steps(self, X, size):
        return np.ceil(len(X)/size).astype(int)

    def reset(self):
        self.loss.reset()
        self.accuracy.reset()

    def batch_data(self, step, size, X, y):
        if size == 1:
            return X, y
        else:
            X_batch = X[step*size:(step+1)*size]
            y_batch = y[step*size:(step+1)*size]
        return X_batch, y_batch
    
    def train(self, X, y, epochs=1, batch_size=1, log=True, log_freq=100, validation=None):
        """Handles the trining loop."""
        # Calculating # of steps for each epoch
        steps = self.get_steps(X, batch_size)
        if validation:
            X_test, y_test = validation
            val_steps = self.get_steps(X_test, batch_size)
        
        for epoch in range(epochs):

            print(f'=== Epoch: {epoch+1} ===')
            self.start_time = time()
            self.reset()
            self.mode_train()

            for step in range(steps):

                # Taking batch
                X_batch, y_batch = self.batch_data(step, batch_size, X, y)

                # Forward Pass
                self.forward(X_batch)

                # Loss 
                self.current_loss = self.get_loss(self.layers[-1].output, y_batch)

                # accuracy 
                self.current_accuracy = self.accuracy.forward(self.layers[-1].output, y_batch) 
                
                # Backward Pass
                self.backward(self.layers[-1].output, y_batch)

                # Optimization 
                self.optim.update(self.trainable_layers)

                # Logging 
                if log:
                    if (not (step % log_freq)) or (step == steps-1):
                        self.step_logger(step, steps)

            self.epoch_logger(epoch, epochs)

            print('--Validation--')
            if validation:
                self.validate(X_test, y_test, batch_size, val_steps, epoch)
                if hasattr(self, 'early_stop_treshold'):
                    if self.check_early_stop():
                        break

    def save(self, path):
        """Save a checkpoint of the model."""
        checkpoint = copy.deepcopy(self)

        # Clearing out data 
        checkpoint.reset()
        # Layer atters to reset 
        l_attributes = ['input', 'output', 'd_w', 'd_b', 'grad']
        for layer in checkpoint.layers:
            for att in l_attributes:
                layer.__dict__.pop(att, None)

        # Saving model
        with open(path, 'wb') as fs: 
            pickle.dump(checkpoint, fs)

    @staticmethod
    def load(path):
        """Load a checkpoint"""
        with open(path, 'rb') as fs:
            checkpoint = pickle.load(fs)
        return checkpoint
