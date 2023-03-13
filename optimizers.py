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
        self.m_weights, self.v_weights = 0, 0
        self.m_biases, self.v_biases = 0, 0

    def update_params(self, weights, biases, delta_weights, delta_biases, epoch):
        epoch += 1
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * delta_weights
        self.m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * delta_biases

        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (delta_weights**2)
        self.v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * (delta_biases**2)

        corrected_m_weights = self.m_weights / (1 - self.beta2 ** epoch)
        corrected_m_biases = self.m_biases / (1 - self.beta2 ** epoch)
        corrected_v_weights = self.v_weights / (1 - self.beta2 ** epoch)
        corrected_v_biases = self.v_biases / (1 - self.beta2 ** epoch)

        weights = weights + self.learning_rate * (corrected_m_weights / (np.sqrt(corrected_v_weights) + self.epsilon))
        # biases = biases - self.learning_rate * (corrected_m_biases / (np.sqrt(corrected_v_biases) + self.epsilon))
        # return weights, biases
        return weights, 0