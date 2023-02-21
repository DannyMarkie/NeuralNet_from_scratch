from layers import Layer, Flatten
from optimizers import Optimizer
from costFunctions import CostFunction
import numpy as np

class NeuralNetwork:
    def __init__(self, layers: Layer):
        self.layers = layers
        self.compiled = False

    def compile(self, optimizer: Optimizer, costFunction: CostFunction):
        self.learningRate = 0.75
        self.optimizer = optimizer
        self.costfunction = costFunction
        self.set_input_sizes()
        self.init_params()
        self.compiled = True

    def fit(self, X, Y, batch_size=1, epochs=3):
        if not self.compiled:
            raise Exception("Model has not been compiled yet. Run model.compile() first.")
        self.gradient_descent(X, Y, batch_size, epochs)

    def add(self, layer: Layer):
        self.compiled = False
        self.layers.append(layer)

    def set_input_sizes(self):
        for layer in self.layers:
            if type(layer) == Flatten:
                previousOutputShape = layer.size
                continue
            layer.inputShape = previousOutputShape
            previousOutputShape = layer.size

    def init_params(self):
        for index, layer in enumerate(self.layers):
            layer.init_params()
            if index == (len(self.layers) - 1):
                self.set_cost_function(layer=layer)
                layer.isLast = True

    def set_cost_function(self, layer):
        layer.activation.add_cost_function(self.costfunction)

    def summary(self):
        if not self.compiled:
            raise Exception("Model has not been compiled yet. Run model.compile() first.")

        for layer in self.layers:
            print(f'{layer}, input shape: {layer.inputShape}, Parameters: {layer.parameterCount}')

    def gradient_descent(self, X, Y, batch_size, iterations):
        sampleSize = X.shape[0]
        for iteration in range(iterations):
            input = X
            for layer in self.layers:
                input = layer.forward_propagation(input=input)
            
            self.layers = np.flip(self.layers)
            deltaZ = 1
            prev_weights = None
            for layer in self.layers:
                if type(layer) == Flatten:
                    continue
                deltaZ, deltaWeights, deltaBiases, prev_weights = layer.backward_propagation(sampleSize, deltaZ, Y, prev_weights)
                layer.update_params(deltaWeights=deltaWeights, deltaBiases=deltaBiases, learningRate=self.learningRate)
            
            self.layers = np.flip(self.layers)

    def predict(self, X):
        next = X
        for layer in self.layers:
            next = layer.feed_forward(input=next)
        return next
    
    def get_accuracy(self, Y_pred, Y_true):
        return np.sum(Y_pred == Y_true) / Y_true.size

    def evaluate(self, X, Y):
        Y_pred = np.argmax(self.predict(X=X), 0)
        print(f'Accuracy: {self.get_accuracy(Y_pred, Y)}')
