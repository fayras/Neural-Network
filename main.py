import numpy
import matplotlib.pyplot

from NeuralNetwork import NeuralNetwork

data_file = open('mnist/mnist_train.csv', 'r')
train_set = data_file.readlines()
data_file.close()

data_file = open('mnist/mnist_test.csv', 'r')
test_set = data_file.readlines()
data_file.close()


def get_data(data_set, index):
    return numpy.asfarray(data_set[index].split(','))


def plot_data(data):
    image_array = data.reshape((28, 28))
    matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
    matplotlib.pyplot.show()
    pass


def scale_data(data):
    return (data / 255.0 * 0.99) + 0.01


def prepare_targets(nodes, digit):
    targets = numpy.zeros(nodes) + 0.01
    targets[int(digit)] = 0.99
    return targets


# plot_data(all_values[1:])

i = 784
h = 64
o = 10
lr = 0.1

epochs = 3

n = NeuralNetwork(i, h, o, lr)

for e in range(epochs):
    for index, entry in enumerate(train_set):
        if index % 5000 == 0:
            print(e, index)
        all_values = numpy.asfarray(entry.split(','))
        inputs = scale_data(all_values[1:])
        targets = prepare_targets(o, all_values[0])
        n.train(inputs, targets)
        pass
    n.set_learning_rate(n.lr / 2.0)
    pass


predictions = []

for entry in test_set:
    all_values = numpy.asfarray(entry.split(','))
    inputs = scale_data(all_values[1:])
    outputs = n.query(inputs)
    correct_label = int(all_values[0])
    prediction = numpy.argmax(outputs)
    if prediction == correct_label:
        predictions.append(1)
    else:
        predictions.append(0)
        pass
    pass

predictions = numpy.asarray(predictions)
print('performance: ', predictions.sum() / predictions.size)