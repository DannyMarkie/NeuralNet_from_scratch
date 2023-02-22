from neuralNetwork import NeuralNetwork
from layers import Flatten, Dense
from activations import ReLU, SoftMax, Tanh
from optimizers import Adam
from costFunctions import CategoricalCrossEntropy, MeanError
from keras.datasets import mnist
from keras.utils import to_categorical

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

Y_train = to_categorical(Y_train)

model = NeuralNetwork([Flatten(inputShape=(28, 28)),
                    #    Dense(size=64, activation=Tanh()),
                       Dense(size=10, activation=Tanh()),
                       Dense(size=10, activation=SoftMax())])

model.compile(Adam(), MeanError())

model.summary()

model.fit(X_train, Y_train, epochs=100)

model.evaluate(X_test, Y_test)