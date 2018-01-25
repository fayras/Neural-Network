import numpy


class NeuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.iNodes = inputnodes
        self.hNodes = hiddennodes
        self.oNodes = outputnodes

        self.lr = learningrate

        self.wih = numpy.random.rand(self.hNodes, self.iNodes) - 0.5
        self.who = numpy.random.rand(self.oNodes, self.hNodes) - 0.5
        pass

    def train(self):
        pass

    def query(self):
        pass

