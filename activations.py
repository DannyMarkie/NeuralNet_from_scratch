import numpy as np
from costFunctions import CostFunction
np.set_printoptions(suppress=True)

class Activation:
    def __init__(self) -> None:
        pass

    def run(self, Z):
        pass

    def deriv(self, Z):
        pass

    def add_cost_function(self):
        pass

class Tanh(Activation):
    def run(self, Z):
        return (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
    
    def deriv(self, Z):
        return 1 - np.square((np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z)))

class ReLU(Activation):
    def run(self, Z):
        return np.maximum(Z, 0)
    
    def deriv(self, Z):
        return Z > 0
    
class SoftMax(Activation):
    def add_cost_function(self, costFunction: CostFunction):
        self.costFunction = costFunction

    def run(self, Z):
        # print(sum(np.exp(Z)))
        A = np.exp(Z) / sum(np.exp(Z))
        # print(A.max())
        return A

    def deriv(self, Z, Y):
        return self.costFunction.run(Y_pred=Z, Y_true=Y)