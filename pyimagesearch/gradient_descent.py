import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def sigmoid_activation(x):
    """
    Calculates the sigmoid for x
    :param x: The value to evaluate
    :return: The sigmoid of x
    """
    return 1.0 / (1 + np.exp(-x))


def predict(X, W):
    """
    Predicts for the data X using the weight W
    :param X: The input data
    :param W: The set of weights
    :return: The predicted values
    """
    # This is the sigma_i w_i * x_i but using all data in X
    preds = sigmoid_activation(X.dot(W))

    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1  # this might be >0.5, but I think it is 0 since prev line has set to 0 some positions
    return preds


(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))
# X = np.c_[np.ones((X.shape[0])), X]
X = np.c_[X, np.ones((X.shape[0]))]

(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

print('[INFO] training...')
W = np.random.randn(X.shape[1], 1)
losses = []

alpha = 0.01
epochs = 100
for epoch in np.arange(0, epochs):
    preds = sigmoid_activation(trainX.dot(W))

    error = preds - trainY
    loss = np.sum(error ** 2)
    losses.append(loss)

    # The gradient descent is the dot product between features and the error of the predictions
    gradient = trainX.T.dot(error)

    # Weights should be in the opposite direction of the gradient and with a factor alpha
    W += -alpha * gradient

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print('[INFO] epoch = {}, loss= {:.7f}'.format(int(epoch + 1), loss))

print('[INFO] evaluating...')
preds = predict(testX, W)
print(classification_report(testY, preds))

# # plot the (testing) classification data

plt.style.use("ggplot")
plt.figure()
plt.title("Data")
# Hack for getting colors to work
color = ['red' if l == 0 else 'green' for l in testY]
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=color, s=30)
# plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY, s=30)
# construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
