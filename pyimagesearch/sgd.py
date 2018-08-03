import numpy as np


def sigmoid_activation(x):
    """
    Computes the sigmoid of x
    :param x: The float to evaluate
    :return: The sigmoid of x
    """
    return 1.0 / (1 + np.exp(-x))
