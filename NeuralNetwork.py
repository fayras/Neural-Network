import numpy
import scipy.special


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

        # Die Aktivierungsfunktion für das Netzwerk
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        targets = numpy.array(targets_list, ndmin=2).T
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lr * numpy.dot(output_errors * final_outputs * (1.0 - final_outputs), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs), numpy.transpose(inputs))

        pass

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def set_learning_rate(self, lr):
        self.lr = lr

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
