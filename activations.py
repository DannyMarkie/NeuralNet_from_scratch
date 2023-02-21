import numpy as np

class Activation:
    def __init__(self) -> None:
        pass

class ReLU(Activation):
    def run(self, Z):
        return np.maximum(Z, 0)
    
    def deriv(self, Z):
        return Z > 0
    
class SoftMax(Activation):
    def run(self, Z):
        pass

    def deriv(self, Z):
        pass