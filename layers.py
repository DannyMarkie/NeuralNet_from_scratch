from activations import Activation
import numpy as np

class Layer:
    def __init__(self, inputShape=None, size=None, activation: Activation=None) -> None:
        self.inputShape = inputShape
        self.size = size
        self.activation = activation

    def init_params(self):
        pass

    def forward_propagation(self, X):
        pass

class Flatten(Layer):
    def __init__(self, inputShape: tuple=None) -> None:
        self.inputShape = inputShape
        self.size = np.prod(inputShape)
        self.weights = []
        self.biases = []
        self.parameterCount = 0

    def __str__(self) -> str:
        return f"Flatten Layer"
    
class Dense(Layer):
    def __init__(self, size, activation: Activation, inputShape=None) -> None:
        super().__init__(inputShape=inputShape, size=size, activation=activation)

    def __str__(self) -> str:
        return f"Dense Layer"
    
    def init_params(self):
        self.weights = np.random.rand(self.size, self.inputShape) - 0.5
        self.biases = np.random.rand(self.size, 1) - 0.5
        self.parameterCount = (len(self.weights) * len(self.weights[0])) + len(self.biases) * len(self.biases[0])
        self.activation.add_cost_function

    def forward_propagation(self, X):
        Z = self.weights.dot(X) + self.biases
        return self.activation.run(Z=Z)
    
    def backward_propagation(self):
        dz = self.activation.deriv()
        pass