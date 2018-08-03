import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def sigmoid_activation(x):
    """
    Computes the sigmoid of x
    :param x: The float to evaluate
    :return: The sigmoid of x
    """
    return 1.0 / (1 + np.exp(-x))


def predict(X, W):
    preds = sigmoid_activation(X.dot(W))

    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1

    return preds


def next_batch(X, y, batch_size):
    for i in np.arange(0, X.shape[0], batch_size):
        yield (X[i:i + batch_size], y[i:i + batch_size])


(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

X = np.c_[X, np.ones((X.shape[0]))]

(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

print('[INFO] training...')
W = np.random.randn(X.shape[1], 1)
losses = []

epochs = 100
alpha = 0.01
batch_size = 32
momentum = 0.9
velocity = 0
for epoch in np.arange(0, epochs):
    epoch_loss = []

    for (batchX, batchY) in next_batch(X, y, batch_size):
        preds = sigmoid_activation(batchX.dot(W))

        error = preds - batchY
        epoch_loss.append(np.sum(error ** 2))

        gradient = batchX.T.dot(error)
        velocity = momentum * velocity - (alpha * gradient)
        W += velocity

        # W += -alpha * gradient

    loss = np.average(epoch_loss)
    losses.append(loss)

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print('[INFO] epoch={}, loss={:.7f}'.format(int(epoch + 1), loss))

print('[INFO] evaluating...')
preds = predict(testX, W)
print(classification_report(testY, preds))

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
