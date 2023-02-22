import numpy as np

class CostFunction:
    def __init__(self) -> None:
        pass

class MeanError(CostFunction):
    def run(self, Y_pred, Y_true):
        # error = np.square((Y_pred - Y_true.T))
        error = (Y_pred - Y_true.T)
        return error

class CategoricalCrossEntropy(CostFunction):
    def run(self, Y_pred, Y_true):
        # return -1 * (Y_true.T * np.log2(Y_pred))
        print(Y_pred)
        # print(-np.sum(Y_true.T * np.log(Y_pred)))
        return -np.multiply(Y_true.T, np.log(Y_pred + 10**-100))

class BinaryCrossEntropy(CostFunction):
    def __init__(self) -> None:
        super().__init__()