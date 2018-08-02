import numpy as np


def line_function(m, x, b):
    return (m * x) + b


def calculate_slope_and_intercept(w0, w1, w2):
    slope = -(w0 / w2) / (w0 / w1)
    # slope = -w1/w2
    intercept = -w0 / w2
    return slope, intercept


def plot_perceptron_and_data(data, weights, labels):
    slope, intercept = calculate_slope_and_intercept(w0=weights[0], w1=weights[1], w2=weights[2])

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    colors = cm.rainbow(np.linspace(0, 1, len(labels)))

    for index, class_label in enumerate(labels):
        x_1 = data[index, 0]
        x_2 = data[index, 1]
        plt.scatter(x_1, x_2, color=colors[class_label])

    frontier_x = list(np.arange(0, 2))
    frontier_y = [line_function(slope, x_i, intercept) for x_i in frontier_x]
    plt.plot(frontier_x, frontier_y, color='magenta', lw=3)
    plt.show()
