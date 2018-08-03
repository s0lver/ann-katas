from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor

image_paths = '...'
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(image_paths, verbose=500)
data = data.reshape((data.shape[0], 3072))

le = LabelEncoder()
labels = le.fit_transform(labels)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=5)

for r in (None, 'l1', 'l2'):
    print('[INFO] training model with "{}" penalty'.format(r))
    model = SGDClassifier(loss='log', penalty=r, max_iter=10, learning_rate='constant', eta0=0.01, random_state=42)
    model.fit(trainX, trainY)

    acc = model.score(testX, testY)
    print('[INFO] "{}" penalty accuracy: {:.2f}%'.format(r, acc * 100))
