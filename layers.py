from activations import Activation
import numpy as np

class Layer:
    def __init__(self, inputShape=None, size=None, activation: Activation=None) -> None:
        self.inputShape = inputShape
        self.size = size
        self.activation = activation

    def init_params(self, optimizer):
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
    
    def init_params(self, optimizer, isLast=False):
        # self.weights = np.random.rand(self.size, self.inputShape) - 0.5
        # self.biases = np.random.rand(self.size, 1) - 0.5
        self.weights = np.random.normal(0.0, pow(self.size, -0.5), (self.size, self.inputShape))
        self.biases = np.random.normal(0.0, pow(self.size, -0.5), (self.size, 1))
        self.moment1 = 0
        self.moment2 = 0
        self.optimizer = optimizer
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
            # deltaZ = self.weights.T.dot(self.activation.deriv(Z=self.next, Y=Y))
            deltaZ = self.activation.deriv(Z=self.next, Y=Y)
            # print(np.sum(deltaZ))
            # print(deltaZ.max())
            # print(np.sum(deltaZ) / sampleSize)
        else: 
            deltaZ = prev_weights.T.dot(deltaZ) * self.activation.deriv(Z=self.Z)
            # deltaZ = prev_weights.T.dot(deltaZ)
            # print(np.sum(deltaZ) / sampleSize)
            # print(np.sum(deltaZ))
        deltaWeights = 1 / sampleSize * self.input.dot((self.next * deltaZ * (1.0 - self.next)).T).T
        # print(self.input.max())
        # deltaWeights = np.dot((deltaZ * self.next * (1.0 - self.next)), self.input.T) / sampleSize
        # print(deltaWeights)
        # print(deltaZ.shape)
        deltaBiases = np.sum(deltaZ) / sampleSize
        # deltaBiases = np.mean(deltaZ, axis=1).reshape(deltaZ.shape[0], 1)
        # deltaBiases = deltaZ
        # print(np.mean(deltaZ, axis=1).shape)
        prev_weights = self.weights
        return deltaZ, deltaWeights, deltaBiases, prev_weights
    
    def update_params(self, deltaWeights, deltaBiases, learningRate, epoch):
        self.weights = self.weights + (learningRate * deltaWeights)
        self.biases = self.biases - (learningRate * deltaBiases)
        # self.weights, self.biases = self.optimizer.update_params(self.weights, self.biases, deltaWeights, deltaBiases, epoch)
