'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
# from sklearn.manifold import locally_linear_embedding
from sklearn.manifold import LocallyLinearEmbedding

np.random.seed(2017)  # for reproducibility

# params
nb_classes = 10
batch_size = 100
nb_epoch = 50
lle_dim = 200
lle_neighbours = 10
dataset_name = "lle_200_dim_mlp_mnist"

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.
X_test /= 255.

# X_train_lle = locally_linear_embedding(X_train, lle_neighbours, lle_dim,
#                                        n_jobs=-1)

# X_test_lle = locally_linear_embedding(X_test, lle_neighbours, lle_dim,
#                                       n_jobs=-1)

# X_train_lle = LocallyLinearEmbedding(lle_neighbours, lle_dim,
#                                      n_jobs=-1).fit_transform(X_train)

# X_test_lle = LocallyLinearEmbedding(lle_neighbours, lle_dim,
#                                     n_jobs=-1).fit_transform(X_test)
# X_valid = X_train_lle[0][:10000]
X_valid = X_train[:10000]
# X_train = X_train_lle[0][10000:]
# X_test = X_test_lle[0]

print(X_train.shape, 'train samples')
print(X_valid.shape, 'valid samples')
print(X_test.shape, 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_valid = Y_train[:10000]
Y_train = Y_train[10000:]
Y_test = np_utils.to_categorical(y_test, nb_classes)

# paper uses sigmoid in the baseline non-LLE DNN
model = Sequential([
    Dense(200, activation='relu', input_shape=(200,)),
    Dropout(0.5),
    Dense(200, activation='relu'),
    Dropout(0.5),
    Dense(100, activation='relu'),
    Dropout(0.5),
    Dense(nb_classes, activation='softmax')
    ])

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-3),
              metrics=['accuracy'])

# best_model = ModelCheckpoint(os.path.join(dataset_name, 'mlp_model_ckpt.h5'),
#                              monitor='val_loss',
#                              save_best_only=True, verbose=1)

history = model.fit(X_train[10000:], Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_valid, Y_valid))
                    # callbacks=[best_model])

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
