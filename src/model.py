"""Module for model creating, loading and saving"""

from layers import Dropout
from utils import accuracy


class Model:
    """Model class designed as a container to simplify the building and training of networks.
    
    Args:
    optimizer:  The optimizer that should be used
    Loss:       The loss class that should be used

    Attributes:
        layers              (list):  List of all layers in the network in their activation sequence 
        trainable_layers    (list):  List of trainable layers in the network
        loss                ():      The loss class that should be used
        optim               ():      The optimizer that should be used
        current_loss        (float): Latest loss recorded 
        current_accuracy    (float): Latest accuracy recorded
        training_mode       (bool):  Boolean flag if the network is in training mode
    """

    def __init__(self, optimizer, loss) -> None:
        self.layers = []
        self.trainable_layers = []
        self.loss = loss
        self.optim = optimizer
        self.current_loss = 0
        self.current_accuracy = 0
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
        
    def logger(self, epoch):
        """Prints current state of model"""
        print(
            f"Epoch: {epoch}, accuracy{self.current_accuracy:.3f}, loss{self.current_loss:.3f}, learning rate {self.optim.clr:.3f} "
        )
    
    def validate(self, X_val, y_val):
        """Handles the validation pass of the network"""
        self.forward(X_val)

        loss = self.loss.forward(self.layers[-1].output, y_val)
        acc  = accuracy(self.layers[-1].output, y_val)

        print(
            f"Validation : Loss: {loss:.3f}, Accuracy: {acc:.3f}"
        )

    def mode_train(self):
        """Sets the model and all dropout layers to training mode."""
        self.training_mode = True
        for l in self.layers:
            if type(l) == Dropout:
                l.training_mode = True
    
    def mode_eval(self):
        """Sets the model and all dropout layers to evaluation mode."""
        self.training_mode = False
        for l in self.layers:
            if type(l) == Dropout:
                l.training_mode = False
    
    def train(self, X, y, epochs=1, log=True, log_freq=100):
        """Handles the trining loop."""
        for epoch in range(epochs + 1):

            # Forward Pass
            self.forward(X)

            # Loss 
            self.current_loss = self.get_loss(self.layers[-1].output, y)

            # accuracy 
            self.current_accuracy = accuracy(self.layers[-1].output, y) 
            
            # Backward Pass
            self.backward(self.layers[-1].output, y)

            # Optimization 
            self.optim.update(self.trainable_layers)

            # Logging 
            if log and not (epoch % log_freq):
                self.logger(epoch)

    