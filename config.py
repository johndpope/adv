from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np

from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.datasets import cifar10, mnist
from sklearn.model_selection import train_test_split


def setup_config():
    """
    Helper function to check correct configuration of tf and keras for tutorial
    :return: True if setup checks completed
    """
    if not hasattr(K, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    # Image dimensions ordering should follow the Theano convention
    if K.image_dim_ordering() != 'tf':
        K.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' "
              "to 'th', temporarily setting to 'tf'")

    # K.set_learning_phase(0)  # without this causes error during learning

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(2017)

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    K.set_session(sess)
    # sess.run(tf.global_variables_initializer())
    return sess


def setup_data(args):
    # Get test data
    if args.dataset == "mnist":
        # trX, trY, teX, teY = data_mnist()
        (trX, trY), (teX, teY) = mnist.load_data()
        x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    elif args.dataset == "cifar":
        (trX, trY), (teX, teY) = cifar10.load_data()
        x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    elif args.dataset == "mnist_lle":
        trX = np.load('data/mnist/trX_lle_10n_784c_mnist.npy')
        trY = np.load('data/mnist/trY_lle_10n_784c_mnist.npy')
        teX = np.load('data/mnist/teX_lle_10n_784c_mnist.npy')
        teY = np.load('data/mnist/teY_lle_10n_784c_mnist.npy')
        x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    elif args.dataset == "cifar_lle":
        trX = np.load('data/cifar10/trX_lle_10n_3072c.npy')
        trY = np.load('data/cifar10/trY_lle_10n_3072c.npy')
        teX = np.load('data/cifar10/teX_lle_10n_3072c.npy')
        teY = np.load('data/cifar10/teY_lle_10n_3072c.npy')
        x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))

    trX = trX.astype('float32')
    teX = teX.astype('float32')
    trX /= 255.
    teX /= 255.
    trY = trY.astype('int32')
    teY = teY.astype('int32')

    if args.split_dataset is not None:
        trX, valX, trY, valY = train_test_split(trX, trY,
                                                test_size=args.split_dataset,
                                                random_state=2017)
    trX = trX.reshape(-1, 28, 28, 1)
    teX = teX.reshape(-1, 28, 28, 1)
    valX = valX.reshape(-1, 28, 28, 1)

    if trY.ndim < 2 and teY.ndim < 2:
        trY = to_categorical(trY, 10)
        teY = to_categorical(teY, 10)
        valY = to_categorical(valY, 10)

    label_smooth = .1
    trY = trY.clip(label_smooth / 9., 1. - label_smooth)
    teY = teY.clip(label_smooth / 9., 1. - label_smooth)
    valY = valY.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    y = tf.placeholder(tf.float32, shape=(None, 10))

    print("trX = {}".format(trX.shape))
    print("trY = {}".format(trY.shape))
    print("valX = {}".format(valX.shape))
    print("valY = {}".format(valY.shape))
    print("teX = {}".format(teX.shape))
    print("teY = {}".format(teY.shape))

    return trX, trY, valX, valY, teX, teY, x, y
