import numpy as np


class Perceptron:
    """
    Class that models the Rosenblatt perceptron
    """

    def __init__(self, input_dimensions, alpha=0.1):
        """
        Creates a Perceptron instance
        :param input_dimensions: The dimensions of the input data (number of features))
        :param alpha: The learning rate
        """
        self.W = np.random.randn(input_dimensions + 1) / np.sqrt(input_dimensions)
        self.alpha = alpha

    @staticmethod
    def step_function(x):
        """
        The step function
        :param x: the input to evaluate
        :return: 1 if x is greater than 0, otherwise 0
        """
        return 1 if x > 0 else 0

    def fit(self, X_train, y, epochs=10) -> None:
        """
        Fits the model for a sample of data
        :param X_train: The input data (matrix)
        :param y: The actual label (classes)
        :param epochs: The number of epochs to train the Perceptron
        :return:
        """

        # This inserts the column for the bias (ones)
        X_train = np.c_[np.ones((X_train.shape[0])), X_train]

        for _ in np.arange(0, epochs):
            for (x_sample, target) in zip(X_train, y):
                # dot product, i.e., sigma w_i * x_i
                weighted_sum = np.dot(x_sample, self.W)
                p = self.step_function(weighted_sum)
                if p != target:
                    error = p - target
                    self.W += -self.alpha * error * x_sample

    def predict(self, X_test):
        # ensure that input is a matrix
        X_test = np.atleast_2d(X_test)

        # This inserts the column for the bias (ones)
        X_test = np.c_[np.ones((X_test.shape[0])), X_test]

        # dot product between the input features and the weight matrix,
        # take such value to the weight matrix
        return self.step_function(np.dot(X_test, self.W))
