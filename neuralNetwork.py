from layers import Layer, Flatten

class NeuralNetwork:
    def __init__(self, layers: Layer):
        self.layers = layers
        self.compiled = False

    def compile(self, optimizer, costFunction):
        self.set_input_sizes()
        self.init_params()
        
        self.compiled = True

    def fit(self, X, Y, batch_size=1, epochs=3):
        if not self.compiled:
            raise Exception("Model has not been compiled yet. Run model.compile() first.")
        pass

    def add(self, layer: Layer):
        self.compiled = False
        self.layers.append(layer)

    def set_input_sizes(self):
        for layer in self.layers:
            if type(layer) == Flatten:
                previousOutputShape = layer.size
                continue
            layer.inputShape = previousOutputShape
            previousOutputShape = layer.size

    def init_params(self):
        for layer in self.layers:
            layer.init_params()

    def summary(self):
        if not self.compiled:
            raise Exception("Model has not been compiled yet. Run model.compile() first.")

        for layer in self.layers:
            print(f'{layer}, input shape: {layer.inputShape}, Parameters: {layer.parameterCount}')