'''Trains a simple convnet on the dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
import os
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from sklearn.manifold import locally_linear_embedding

np.random.seed(2017)  # for reproducibility

# params
nb_classes = 10
batch_size = 500
nb_epoch = 50
lle_dim = 432  # 12x12x3
lle_neighbours = 40
# input image dimensions
img_rows, img_cols = 12, 12
# size of pooling area for max pooling
pool_size = (2, 2)
# dataset name
dataset_name = "lle_432_dim_cnn_cifar10"

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.reshape(X_train.shape[0], np.prod(X_train.shape[1:]))
X_test = X_test.reshape(X_test.shape[0], np.prod(X_test.shape[1:]))
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.
X_test /= 255.

X_train_lle = locally_linear_embedding(X_train, lle_neighbours, lle_dim,
                                       n_jobs=-1)
X_test_lle = locally_linear_embedding(X_test, lle_neighbours, lle_dim,
                                      n_jobs=-1)

X_valid = X_train_lle[0][:10000]
X_train = X_train_lle[0][10000:]
X_test = X_test_lle[0]

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    X_valid = X_valid.reshape(X_valid.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

print(X_train.shape, 'train samples')
print(X_valid.shape, 'valid samples')
print(X_test.shape, 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_valid = Y_train[:10000]
Y_train = Y_train[10000:]
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential([
    Convolution2D(32, 8, 8, activation='relu', border_mode='same',
                  input_shape=input_shape),
    Convolution2D(32, 4, 4, activation='relu', border_mode='same'),
    MaxPooling2D(pool_size=pool_size),
    Flatten(),
    Dense(144, activation='relu'),
    Dropout(0.25),
    Dense(72, activation='relu'),
    Dropout(0.25),
    Dense(36, activation='relu'),
    Dropout(0.25),
    Dense(nb_classes, activation='softmax'),
    ])

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=1e-3),
              metrics=['accuracy'])

best_model = ModelCheckpoint(os.path.join(dataset_name, 'cnn_model_ckpt.h5'),
                             monitor='val_loss',
                             verbose=1, save_best_only=True)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_valid, Y_valid),
          callbacks=[best_model])

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
