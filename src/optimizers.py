"""Module for optimization functions"""

class SDG:
    """Stochastic Gradient Decent class used to update layer paramers
    The update is the -ve learning rate multiplied by the gradient calculated in the backward step.

    Attr:
        lr (float) Learning rate to scale the gradients by for the update
    """
    IMPLEMENTED = [LinearLayer]

    def __init__(self, learning_rate=1, decay=0., momentum=0.) -> None:
        self.lr = learning_rate
        self.clr = learning_rate # current learning rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0

    def init_momentum(self, layers):
        for layer in layers:
            if not hasattr(layer, 'momentum_w'):
                layer.momentum_w = np.zeros_like(layer.weights)
                layer.momentum_b = np.zeros_like(layer.bias)

    def pre_update_step(self):
        decay_rate = 1/(1 + self.decay * self.iterations)
        self.clr = self.lr * decay_rate

    def get_updates(self, layer):
        return (
            -self.clr*layer.d_w,
            -self.clr*layer.d_b
        )

    def get_momentum_updates(self, layer):
        wu = (self.momentum * layer.momentum_w) - (self.clr * layer.d_w) 
        bu = (self.momentum * layer.momentum_b) - (self.clr * layer.d_b) 
        layer.momentum_w = wu
        layer.momentum_b = bu
        return (wu, bu)

    def update(self, layers):
        """Update a layers parameters.
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