import numpy as np

class CostFunction:
    def __init__(self) -> None:
        pass

class MeanSquaredError(CostFunction):
    def run(self, Y_pred, Y_true):
        return Y_pred - Y_true.T

class CategoricalCrossEntropy(CostFunction):
    def __init__(self) -> None:
        super().__init__()

class BinaryCrossEntropy(CostFunction):
    def __init__(self) -> None:
        super().__init__()