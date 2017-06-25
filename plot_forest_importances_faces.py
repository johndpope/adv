"""
=================================================
Pixel importances with a parallel forest of trees
=================================================

This example shows the use of forests of trees to evaluate the importance
of the pixels in an image classification task (faces). The hotter the pixel,
the more important.

The code below also illustrates how the construction and the computation
of the predictions can be parallelized within multiple jobs.
"""

from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.ensemble import ExtraTreesClassifier
from keras.datasets import mnist
import numpy as np

print(__doc__)

# Number of cores to use to perform parallel fitting of the forest model
n_jobs = -1

# Load the faces dataset
(trX, trY), (teX, teY) = mnist.load_data()
trX = np.float32(trX / 255.)
trX = trX.reshape(-1, np.prod(trX.shape[1:]))
teX = np.float32(teX / 255.)
teX = teX.reshape(-1, np.prod(teX.shape[1:]))
# data = fetch_olivetti_faces()
# X = data.images.reshape((len(data.images), -1))
# y = data.target

mask = trY < 5  # Limit to 5 classes
trX = trX[mask]
trY = trY[mask]

# Build a forest and compute the pixel importances
print("Fitting ExtraTreesClassifier on faces data with %d cores..." % n_jobs)
t0 = time()
forest = ExtraTreesClassifier(n_estimators=1000,
                              max_features=128,
                              n_jobs=n_jobs,
                              random_state=0)

forest.fit(trX, trY)
print("done in %0.3fs" % (time() - t0))
importances = forest.feature_importances_
importances = importances.reshape(28, 28)

# Plot pixel importances
plt.matshow(importances, cmap=plt.cm.hot)
plt.title("Pixel importances with forests of trees")
plt.show()
