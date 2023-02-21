from neuralNetwork import NeuralNetwork
from layers import Flatten, Dense
from activations import ReLU, SoftMax
from optimizers import Adam
from costFunctions import CategoricalCrossEntropy

model = NeuralNetwork([Flatten(inputShape=(28, 28)),
                       Dense(size=128, activation=ReLU()),
                       Dense(size=10, activation=SoftMax())])

model.compile(Adam(), CategoricalCrossEntropy())

model.summary()