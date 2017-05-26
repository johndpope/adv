'''Create a convolutional neural network for compressing the input images.
'''

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import SGD
from scipy.spatial import distance
import sklearn.preprocessing
import numpy as np


def create_cnn():
    """Create a convolutional neural network for compressing the input images.

    Reference:
    Koutnik, Jan, Jurgen Schmidhuber, and Faustino Gomez. "Evolving deep
    unsupervised convolutional networks for vision-based reinforcement
    learning." Proceedings of the 2014 conference on Genetic and
    evolutionary computation. ACM, 2014.
    """
    model = Sequential()

    model.add(Conv2D(64, (8, 8), strides=(2, 2), activation='relu',
                     padding='same', input_shape=(28, 28, 1)))
    #model.add(MaxPool2D(pool_size=(3, 3)))

    model.add(Conv2D(128, (6, 6), strides=(2, 2), activation='relu',
                     padding='valid'))
    #model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (5, 5), strides=(1, 1),
                     activation='relu', padding='valid'))
    #model.add(MaxPool2D(pool_size=(3, 3)))

    #model.add(Conv2D(3, (2, 2),
    #                 activation='relu', input_shape=(3, 3, 10)))
    #model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    # The model needs to be compiled before it can be used for prediction
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    return model


def calculate_cnn_output(model, input):
    output = model.predict(input)
    # output = output.reshape(output.shape[0], output.shape[1])
    output = output.reshape(-1, np.prod(output.shape[1:]))

    normalized_output = sklearn.preprocessing.normalize(output)

    return normalized_output


def calculate_fitness(feature_vectors):
    pairwise_euclidean_distances = distance.pdist(feature_vectors, 'euclidean')
    fitness = pairwise_euclidean_distances.mean() + \
        pairwise_euclidean_distances.min()
    return fitness
