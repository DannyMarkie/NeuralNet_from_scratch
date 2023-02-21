import numpy as np
from costFunctions import CostFunction

class Activation:
    def __init__(self) -> None:
        pass

    def run(self, Z):
        pass

    def deriv(self, Z):
        pass

    def add_cost_function(self):
        pass

class ReLU(Activation):
    def run(self, Z):
        return np.maximum(Z, 0)
    
    def deriv(self, Z):
        return Z > 0
    
class SoftMax(Activation):
    def add_cost_function(self, costFunction: CostFunction):
        self.costFunction = costFunction

    def run(self, Z):
        return (np.exp(Z) / sum(np.exp(Z)))

    def deriv(self, Z):
        return CostFunction.run()