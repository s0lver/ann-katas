import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn import datasets
from sklearn.externals.six.moves import urllib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

print("[INFO] loading MNIST (full) dataset...")


def get_dataset():
    try:
        from sklearn.datasets import fetch_mldata
        mnist = fetch_mldata('MNIST original')
    except urllib.error.HTTPError as ex:
        print("Could not download MNIST data from mldata.org, trying alternative...")

        # Alternative method to load MNIST, if mldata.org is down
        from scipy.io import loadmat
        mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
        mnist_path = "./mnist-original.mat"
        response = urllib.request.urlopen(mnist_alternative_url)
        with open(mnist_path, "wb") as f:
            content = response.read()
            f.write(content)
        mnist_raw = loadmat(mnist_path)
        mnist = {
            "data": mnist_raw["data"].T,
            "target": mnist_raw["label"][0],
            "COL_NAMES": ["label", "data"],
            "DESCR": "mldata.org dataset: mnist-original",
        }
        print("Success!")

        return mnist


# dataset = get_dataset()
from scipy.io import loadmat

mnist_raw = loadmat("./mnist-original.mat")
dataset = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0],
    "COL_NAMES": ["label", "data"],
    "DESCR": "mldata.org dataset: mnist-original",
}

# data = dataset.data.astype("float") / 255.0
data = dataset["data"].astype("float") / 255.0
# (trainX, testX, trainY, testY) = train_test_split(data, dataset.target, test_size=0.25)
(trainX, testX, trainY, testY) = train_test_split(data, dataset["target"], test_size=0.25)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# architecture 784-256-28-10 (784 is bc the 28*28 image size)
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

print("[INFO] training network using SGD")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
# Note that validation data could be passed on, here the same test data is being used
# H is the training loss and metric values at each epoch
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128)

# evaluating the network
print("[INFO] evaluating network")
predictions = model.predict(testX, batch_size=128)
print(classification_report(
    testY.argmax(axis=1), predictions.argmax(axis=1),
    target_names=[str(x) for x in lb.classes_]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
# plt.savefig(args["output"])
