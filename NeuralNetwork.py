import numpy


class NeuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.iNodes = inputnodes
        self.hNodes = hiddennodes
        self.oNodes = outputnodes

        # Die Lernrate, wie schnell die Gewichte angepasst werden.
        self.lr = learningrate

        # Die Gewichte des Netzwerks zwischen den einzelnen
        # Knotenschichten.
        self.wih = NeuralNetwork.normal_weights(self.hNodes, self.iNodes)
        self.who = NeuralNetwork.random_weights(self.oNodes, self.hNodes)
        pass

    def train(self):
        pass

    def query(self):
        pass

    @staticmethod
    def random_weights(rows, cols):

        # Liefert komplett zufällige Gewichte zurück.
        # Diese befinden sind zwischen -0.5 und 0.5

        return numpy.random.rand(rows, cols) - 0.5

    @staticmethod
    def normal_weights(rows, cols):

        # Liefert eine Matrix der Größe mit rows x cols.
        # Dabei werden die Gewichte nicht zufällig gewählt,
        # sondern werden nach einer Normalverteilung gewählt.

        return numpy.random.normal(0.0, pow(rows, -0.5), (rows, cols))

