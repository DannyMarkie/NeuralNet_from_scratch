from neuralNetwork import NeuralNetwork
from layers import Flatten, Dense
from activations import ReLU, SoftMax
from optimizers import Adam
from costFunctions import CategoricalCrossEntropy
from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

model = NeuralNetwork([Flatten(inputShape=(28, 28)),
                       Dense(size=128, activation=ReLU()),
                       Dense(size=10, activation=SoftMax())])

model.compile(Adam(), CategoricalCrossEntropy())

model.summary()

model.fit(X_train, Y_train)