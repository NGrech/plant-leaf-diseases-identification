"""Module for model creating, loading and saving"""
import numpy as np
from layers import Dropout
from loss import Accuracy


class Model:
    """Model class designed as a container to simplify the building and training of networks.
    
    Args:
    optimizer:  The optimizer that should be used
    Loss:       The loss class that should be used

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
        self.accuracy = Accuracy
        self.optim = optimizer
        self.current_loss = 0
        self.training_mode = True

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
        """Prints current state of model"""
        print(
            f"Step: {step}/{steps}, accuracy{self.accuracy.current_accuracy:.3f}, loss{self.current_loss:.3f}, learning rate {self.optim.clr:.3f} "
        )

    def epoch_logger(self, epoch, epochs):
        """Prints current state of model"""
        print(
            f"Epoch: {epoch}/{epochs}, accuracy{self.accuracy.get_accumulated_accuracy():.3f}, loss{self.loss.get_accumulated_loss():.3f}, learning rate {self.optim.clr:.3f}"
        )
    
    def validate(self, X_val, y_val, batch_size, steps):
        """Handles the validation pass of the network"""
        self.mode_eval()
        self.reset()

        for step in range(steps):
            X_batch, y_batch = self.batch_data(step, batch_size, X_val, y_val)

            self.forward(X_batch)

            self.loss.forward(self.layers[-1].output, y_batch)
            self.accuracy.forward(self.layers[-1].output, y_batch)

        print(
            f"Validation : Accuracy: {self.accuracy.get_accumulated_accuracy():.3f}, Loss: {self.loss.get_accumulated_loss():.3f}"
        )

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
        return np.ceil(len(X)/size)

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
        
        for epoch in range(epochs + 1):

            print(f'=== Epoch: {epoch+1} of {epochs} ===')

            self.reset()
            self.training_mode()

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

            if validation:
                self.validate(X_test, y_test, batch_size, val_steps)

    