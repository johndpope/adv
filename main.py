from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import argparse
import tensorflow as tf
import keras.backend as K
# from keras.datasets import mnist
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.utils.visualize_util import plot
from sklearn.model_selection import train_test_split

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation
from cleverhans.utils import cnn_model, pair_visual, grid_visual
from models import hierarchical, irnn, mlp, siamese, identity_model
from models import mlp_lle, cnn_lle, cnn_model
from utils import rank_classifiers, rank_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for "
                                     "adversarial samples.")
    parser.add_argument("-b", "--batch_size", type=int, default=128,
                        help="Batch size for training "
                        "and testing the model")
    parser.add_argument("-e", "--epochs", type=int, default=15,
                        help="Nb of epochs for training")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1,
                        help="Nb of epochs for training")
    parser.add_argument("-m", "--model", type=str, default="cnn_model",
                        help="The model used for taining and"
                        "testing against adversarial attacks")
    parser.add_argument("-a", "--attack", type=str, default="fgsm",
                        help="Choose the method of attack, "
                        "1.)fgsm, 2.)jmsa")
    parser.add_argument("dataset", type=str, default="mnist",
                        help="Choose the dataset to be used for "
                        "the training and the attacks")
    parser.add_argument("-c", "--nb_classes", type=int, default=10,
                        help="Choose the number of classes in your"
                        "dataset")
    parser.add_argument("-rc", "--rank_classifiers", type=bool, default=False,
                        help="Rank classifiers based on their accuracy")
    parser.add_argument("-rf", "--rank_features", type=bool, default=False,
                        help="Rank feature importance using xgboost")
    parser.add_argument("-p", "--plot_arch", type=bool, default=False,
                        help="Rank classifiers based on their accuracy")
    parser.add_argument("-s", "--split_dataset", type=float, default=0.2,
                        help="Rank classifiers based on their accuracy")
    parser.add_argument("-eps", "--eps", type=float, default=0.3,
                        help="Epsilon variable for adversarial distortion")
    parser.add_argument("-pv", "--pair_visual", type=int, default=5,
                        help="Plot normal and distorted image from "
                             "particular target classs")
    parser.add_argument("-gv", "--grid_visual", type=bool, default=False,
                        help="Plot normal and distorted image from "
                             "particular target classs")
    args = parser.parse_args()

    K.set_learning_phase(0)  # without this causes error during learning

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(2017)

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    K.set_session(sess)
    # sess.run(tf.global_variables_initializer())

    # Get test data
    if args.dataset == "mnist":
        X, Y, X_test, Y_test = data_mnist()
        x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    elif args.dataset == "cifar":
        X, Y, X_test, Y_test = cifar10()
        x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    elif args.dataset == "mnist_lle":
        X = np.load('trX_lle_10n_200c_mnist.npy')
        Y = np.load('trY_lle_10n_200c_mnist.npy')
        X_test = X[60000:]
        Y_test = Y[60000:]
        X = np.delete(X, X[60000:], axis=1)
        Y = np.delete(Y, Y[60000:], axis=1)
        x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    elif args.dataset == "cifar_lle":
        x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))

    if args.split_dataset is not None:
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y,
                                                          test_size=args.split_dataset,
                                                          random_state=2017)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    if Y_train.ndim < 2 and Y_test.ndim < 2:
        Y_train = np_utils.to_categorical(Y_train, 10)
        Y_test = np_utils.to_categorical(Y_test, 10)
        Y_val = np_utils.to_categorical(Y_val, 10)

    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Define TF model graph
    model = eval(args.model + '()')
    model.summary()
    predictions = model(x)
    print("Defined TensorFlow graph.")

    if args.plot_arch is True:
        plot(model, to_file='identity_model.png', show_shapes=True,
             show_layer_names=True)

    if args.rank_features is True:
        rank_features(X.reshape(-1, 784), np.argmax(Y, axis=1))

    train_params = {'nb_epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate}
    eval_params = {'batch_size': args.batch_size}

    def evaluate_legit():
        accuracy = model_eval(sess, x, y, predictions,
                              X, Y, args=eval_params)
        print("Test accuracy on legitimate test examples: {}"
              .format(accuracy))
        return accuracy

    # train on dataset
    model_train(sess, x, y, predictions, X_train, Y_train,
                evaluate=evaluate_legit, args=train_params)

    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    fgsm = FastGradientMethod(model, sess=sess)
    fgsm_params = {'eps': args.eps}
    adv_x = fgsm.generate(x, **fgsm_params)
    preds_adv = model(adv_x)
    X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test],
                             args=eval_params)
    assert X_test_adv.shape[0] == 10000, X_test_adv.shape

    if args.pair_visual is not None:
        pair_visual(X_test[args.pair_visual].reshape(28, 28),
                    X_test_adv[args.pair_visual].reshape(28, 28))

    if args.grid_visual is True:
        if args.dataset == "mnist":
            labels = np.unique(np.argmax(Y_train, axis=1))
            data = X_train[labels]
        else:
            labels = np.unique(np.argmax(Y_test, axis=1))
            data = X_test[labels]
        grid_visual(np.hstack((labels, data)))

    if args.rank_classifiers is True:
        cnn = cnn_model()
        per = mlp()
        hr = hierarchical()
        ir = irnn()
        idd = identity_model()
        models = [("cnn_model", cnn),
                  ("mlp", per),
                  ("hierarchical", hr),
                  ("irnn", ir),
                  ("identity_model", idd)]

        rank_classifiers(models, X_train, Y_train, X_test, X_test_adv,
                         Y_test, epochs=args.epochs,
                         batch_size=args.batch_size)

    # Evaluate the accuracy of the MNIST model on adversarial examples
    # X_test_adv might change to X_test as in the new cleverhans example
    adv1_accuracy = model_eval(sess, x, y, preds_adv, X_test_adv, Y_test,
                               args=eval_params)
    print("Test accuracy on adversarial examples: ".format(adv1_accuracy))

    print("Repeating the process, using aversarial training")
    # Redefine TF model graph
    model_2 = eval(args.model + '()')
    predictions_2 = model_2(x)
    fgsm2 = FastGradientMethod(model_2, sess=sess)
    adv_x_2 = fgsm.generate(x, **fgsm_params)
    predictions_2_adv = model_2(adv_x_2)

    def evaluate_adversarial():
        # Evaluate the accuracy of the adversarialy trained MNIST model on
        # legitimate test examples
        adv2_accuracy = model_eval(sess, x, y, predictions_2, X_test, Y_test,
                                   args=eval_params)
        print("Test accuracy on legitimate test examples: {}"
              .format(adv2_accuracy))

        # Evaluate the accuracy of the adversarially trained MNIST model on
        # adversarial examples
        adv3_accuracy = model_eval(sess, x, y, predictions_2_adv, X_test,
                                   Y_test, args=eval_params)
        print("Test accuracy on adversarial examples: {}"
              .format(adv3_accuracy))

    # Perform adversarial training
    model_train(sess, x, y, predictions_2, X_train, Y_train,
                predictions_adv=predictions_2_adv,
                evaluate=evaluate_adversarial, args=train_params)
