import numpy as np


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.W = []
        self.layers = layers
        self.alpha = alpha
        self.losses = []  # for plotting purposes
        self._initialize_weights()

    def _initialize_weights(self):
        for i in np.arange(0, len(self.layers) - 2):
            # The +1 is for the bias in initial layers
            w = np.random.randn(self.layers[i] + 1, self.layers[i + 1] + 1)
            # scale w by dividing by the square root of the number of nodes in the current layer, thereby
            # normalizing the variance of each neuron’s output
            self.W.append(w / np.sqrt(self.layers[i]))

        # no bias for output layer
        w = np.random.randn(self.layers[-2] + 1, self.layers[-1])
        self.W.append(w / np.sqrt(self.layers[-2]))

    def __repr__(self):
        return 'NeuralNetwork: {}'.format('-'.join(str(l) for l in self.layers))

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def fit_partial(self, x, y):
        A = [np.atleast_2d(x)]

        # Feedforward
        for layer in np.arange(0, len(self.W)):
            # RECALL: VECTRIZED
            # dot product between activation and weight matrix, aka net_input
            net = A[layer].dot(self.W[layer])

            # net_output
            out = self.sigmoid(net)

            A.append(out)

        # Back-propagation
        # 1. Compute the difference between prediction (final output activation) and true target
        error = A[-1] - y

        # 2. Apply chain rule and build list of delta 'D';
        # first entry is the error of the output layer times the derivative of the activation function
        D = [error * self.sigmoid_derivative(A[-1])]

        for layer in np.arange(len(A) - 2, 0, -1):
            # The delta for the current layer is the delta of *previous* layer dotted with the weight matrix of
            # the current layer, followed by multiplying the delta by the derivative of the activation function
            # for the activations of current layer
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_derivative(A[layer])
            D.append(delta)

        # Reversing the deltas since we worked backwards
        D = D[::-1]
        for layer in np.arange(0, len(self.W)):
            # update weights by taking dot product of layer activations with respective deltas abd multiplying it
            # by learning rate
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def fit(self, X, y, epochs=1000, display_update_every=100):
        # X = np.c_[np.ones(X.shape[0]), X]
        X = np.c_[X, (np.ones(X.shape[0]))]

        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            if epoch == 0 or (epoch + 1) % display_update_every == 0:
                loss = self.calculate_loss(X, y)
                self.losses.append(loss)
                print('[INFO] epoch={}, loss={:.7f}'.format(epoch + 1, loss))

    def predict(self, X, add_bias=True):
        p = np.atleast_2d(X)

        if add_bias:
            # p = np.c_[np.ones(p.shape[0]), p]
            p = np.c_[p, (np.ones(p.shape[0]))]

        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p

    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, add_bias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)
        return loss
