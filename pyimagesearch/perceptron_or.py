import numpy as np

from pyimagesearch.nn.perceptron import Perceptron
from pyimagesearch.nn.perceptron_plots import plot_perceptron_and_data

if __name__ == '__main__':
    # this is the OR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [1]])

    print('Creating Perceptron')
    p = Perceptron(X.shape[1], alpha=0.1)

    print('Training Perceptron')
    p.fit(X, y, epochs=20)

    print('Testing Perceptron')
    for (x, target) in zip(X, y):
        predicted = p.predict(x)
        print("data={}, ground-truth={}, pred={}".format(x, target[0], predicted))

print('\nWeights were {}'.format(p.W))
plot_perceptron_and_data(X, p.W, y)
