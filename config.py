# coding: utf-8

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
from deap import base, creator, tools
from utils import calculate_fitness, update_model_weights
from utils import calculate_model_output


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
        trX = np.load('test/trX_lle_all_mnist.npy')
        trY = np.load('test/trY_lle_all_mnist.npy')
        teX = np.load('test/teX_lle_all_mnist.npy')
        teY = np.load('test/teY_lle_all_mnist.npy')
        x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    elif args.dataset == "cifar_lle":
        trX = np.load('data/cifar10/trX_lle_10n_3072c.npy')
        trY = np.load('data/cifar10/trY_lle_10n_3072c.npy')
        teX = np.load('data/cifar10/teX_lle_10n_3072c.npy')
        teY = np.load('data/cifar10/teY_lle_10n_3072c.npy')
        x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))

    trX = trX.astype('float32')
    teX = teX.astype('float32')
    # trX /= 255.
    # teX /= 255.
    trY = trY.astype('int32')
    teY = teY.astype('int32')

    if args.split_dataset is not None:
        trX, valX, trY, valY = train_test_split(trX, trY,
                                                test_size=args.split_dataset,
                                                random_state=2017)
    if args.dataset == "mnist" or args.dataset == "mnist_lle":
        trX = trX.reshape(-1, 28, 28, 1)
        teX = teX.reshape(-1, 28, 28, 1)
        valX = valX.reshape(-1, 28, 28, 1)
    if args.dataset == "cifar" or args.dataset == "cifar_lle":
        trX = trX.reshape(-1, 32, 32, 3)
        teX = teX.reshape(-1, 32, 32, 3)
        valX = valX.reshape(-1, 32, 32, 3)

    if trY.ndim <= 2 and teY.ndim <= 2:
        trY = to_categorical(trY, 10)
        teY = to_categorical(teY, 10)
        valY = to_categorical(valY, 10)

    label_smooth = .1
    trY = trY.clip(label_smooth / 9., 1. - label_smooth)
    teY = teY.clip(label_smooth / 9., 1. - label_smooth)
    valY = valY.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # print("trX = {}".format(trX.shape))
    # print("trY = {}".format(trY.shape))
    # print("valX = {}".format(valX.shape))
    # print("valY = {}".format(valY.shape))
    # print("teX = {}".format(teX.shape))
    # print("teY = {}".format(teY.shape))

    return trX, trY, valX, valY, teX, teY, x, y


def ga_fitness(individual, model, data):
    # The GA will update the RNN parameters to evaluate candidate solutions
    update_model_weights(model, np.asarray(individual))

    # Calculate the output feature vectors
    feature_vectors = calculate_model_output(model, data)

    # Check their fitness
    fitness = calculate_fitness(feature_vectors)

    return fitness


def ga_setup(model, data):
    # Set up the genetic algorithm to evolve the RNN parameters
    # creates a FitnessMin class for a minimization problem
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # craetes an Individual class that is derived from a list with a
    # fitness attribute set to the created fitness class
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Once the types are created you need to fill them with sometimes
    # random values, sometime guessed ones. DEAP provides an easy mechanism
    # to do that. The Toolbox is a container for tools of all sorts
    # including initializers that can do what is needed of them.
    # The following takes on the last lines of code to create the initializers
    # for individuals containing random floating point numbers and for a
    # population that contains them.
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, -1.5, 1.5)
    toolbox.register("individual",
                     tools.initRepeat,
                     creator.Individual,
                     toolbox.attr_float,
                     n=model.count_params())  # total #params of CNN
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def ga_fitness(individual):
        # The GA will update the RNN parameters to evaluate candidate solutions
        update_model_weights(model, np.asarray(individual))

        # Calculate the output feature vectors
        feature_vectors = calculate_model_output(model, data)

        # Check their fitness
        fitness = calculate_fitness(feature_vectors)

        return fitness,

    # Operators are just like initializers, except that some are already
    # implemented in the tools module. Once you’ve chosen the perfect ones,
    # simply register them in the toolbox. In addition you must create your
    # evaluation function.
    toolbox.register("evaluate", ga_fitness)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.5,
                     indpb=0.10)

    # Use tournament selection: choose a subset consisting of
    # k members of that
    # population, and from that subset, choose the best individual
    toolbox.register("select", tools.selTournament,
                     tournsize=10)  # optimize this hyperparameter

    return toolbox
