from activations import Activation
import numpy as np

class Layer:
    def __init__(self, inputShape=None, size=None, activation: Activation=None) -> None:
        self.inputShape = inputShape
        self.size = size
        self.activation = activation

    def init_params(self):
        pass

    def forward_propagation(self, input):
        pass

    def backward_propagation(self, sampleSize):
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
    
    def forward_propagation(self, input):
        return input.reshape(input.shape[0], -1).T
    
    def feed_forward(self, input):
        return self.forward_propagation(input=input)
    
class Dense(Layer):
    def __init__(self, size, activation: Activation, inputShape=None) -> None:
        super().__init__(inputShape=inputShape, size=size, activation=activation)

    def __str__(self) -> str:
        return f"Dense Layer"
    
    def init_params(self, isLast=False):
        self.weights = np.random.rand(self.size, self.inputShape) - 0.5
        self.biases = np.random.rand(self.size, 1) - 0.5
        self.parameterCount = (len(self.weights) * len(self.weights[0])) + len(self.biases) * len(self.biases[0])
        self.isLast = isLast

    def forward_propagation(self, input):
        self.input = input
        self.Z = self.weights.dot(self.input) + self.biases
        self.next = self.activation.run(Z=self.Z)
        return self.next
    
    def feed_forward(self, input):
        Z = self.weights.dot(input) + self.biases
        return self.activation.run(Z=Z)
    
    def backward_propagation(self, sampleSize, deltaZ, Y, prev_weights):
        if self.isLast:
            deltaZ = self.activation.deriv(Z=self.next, Y=Y)
        else: 
            deltaZ = prev_weights.T.dot(deltaZ) * self.activation.deriv(Z=self.Z)
        deltaWeights = 1 / sampleSize * deltaZ.dot(self.input.T)
        # print(deltaZ.shape)
        # deltaBiases = 1 / sampleSize * np.sum(deltaZ)
        print(deltaZ[0])
        deltaBiases = np.sum(deltaZ[0])
        prev_weights = self.weights
        # print(deltaWeights.shape)
        # print(f'{self.size}: {deltaBiases.max(), deltaBiases.min()}')
        return deltaZ, deltaWeights, deltaBiases, prev_weights
    
    def update_params(self, deltaWeights, deltaBiases, learningRate):
        # print(self.biases.shape, deltaBiases)
        self.weights = self.weights - (learningRate * deltaWeights)
        self.biases = self.biases - (learningRate * deltaBiases)
        # print(self.biases.shape)
        # print(f'{self.biases, (learningRate * deltaBiases), deltaBiases}')