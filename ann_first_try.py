# Trying to do a network with a hidden layer
# 2 neurons and the bias
import numpy as np

X_raw = [[0,0], [0,1], [1,0], [1,1]]
Y_raw = [0,1,1,0]

weights = np.ones(4)
neurons = ...


def feed_forward():
	X = np.array(X_raw)
	Y = np.array(Y_raw)
	print(X)
	print(Y)


if __name__ == '__main__':
	feed_forward()
