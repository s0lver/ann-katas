import numpy as np
import matplotlib.pyplot as plt

from pyimagesearch.nn.NeuralNetwork import NeuralNetwork

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork([2, 2, 1], alpha=0.5)
nn.fit(X, y, epochs=20000)

# nn = NeuralNetwork([2, 1], alpha=0.5)
# nn.fit(X, y, epochs=20000)

for (x, target) in zip(X, y):
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print('[INFO] data={}, ground_truth={}, pred={:.4f}, step={}'.format(x, target[0], pred, step))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(nn.losses)), nn.losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
