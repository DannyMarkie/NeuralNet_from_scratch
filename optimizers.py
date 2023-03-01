import numpy as np

class Optimizer:
    def __init__(self) -> None:
        pass

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=10E-8) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def update_moment(self, input, deriv):
        for i in range(input):
            self.moment1[i] = self.beta1 * self.moment1[i] + (1.0 - self.beta1) * deriv
            self.moment2[i] = self.beta2 * self.moment2[i] + (1.0 - self.beta2) * deriv**2