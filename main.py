from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import argparse
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.utils import np_utils
# from keras.utils.visualize_util import plot
from sklearn.model_selection import train_test_split

# from cleverhans.utils_mnist import data_mnist, data_cifar
# from cleverhans.utils_tf import model_train, model_eval, batch_eval
# from cleverhans.attacks import fgsm
# from cleverhans.utils import cnn_model, pair_visual, grid_visual
from models import hierarchical, irnn, mlp, siamese, identity_model
from models import mlp_lle, cnn_lle, cnn_model
from utils import rank_classifiers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for "
                                     "adversarial samples.")
    parser.add_argument("-b", "--batch_size", type=int, default=128,
                        help="Batch size for training "
                        "and testing the model")
    parser.add_argument("-e", "--epochs", type=int, default=15,
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
    args = parser.parse_args()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # keras.backend.learning_phase() = tf.constant(0) # test mode

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Get test data
    if args.dataset == "mnist":
        (X, Y), (X_test, Y_test) = mnist.load_data()
        # X, Y, X_test, Y_test = data_mnist()
    elif args.dataset == "cifar":
        X, Y, X_test, Y_test = data_cifar()
    elif args.dataset == "mnist_lle":
        X = np.load('trX_lle_10n_200c_mnist.npy')
        Y = np.load('trY_lle_10n_200c_mnist.npy')
        X_test = X[60000:]
        Y_test = Y[60000:]
        X = np.delete(X, X[60000:], axis=1)
        Y = np.delete(Y, Y[60000:], axis=1)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y,
                                                      test_size=0.15,
                                                      random_state=42)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    if Y_train.ndim < 2 and Y_test.ndim < 2:
        Y_train = np_utils.to_categorical(Y_train, 10)
        Y_test = np_utils.to_categorical(Y_test, 10)
        Y_val = np_utils.to_categorical(Y_val, 10)

    # label_smooth = .1
    # Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

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

    rank_classifiers(models, X_train, Y_train)
    import pdb
    pdb.set_trace()
    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Define TF model graph
    model = eval(args.model + '()')
    model.summary()
    predictions = model(x)
    # model.fit(X_train, Y_train, nb_epoch=1000, batch_size=500, shuffle=True,
    # validation_data=(X_val, Y_val), verbose=1)
    print("Defined TensorFlow cnn model graph.")

    # plot(identity, to_file='identity_model.png', show_shapes=True,
    #      show_layer_names=True)

    train_params = {'nb_epochs': args.epoc,
                    'batch_size': args.batch_size,
                    'learning_rate': 1e-3}
    eval_params = {'batch_size': args.batch_size}
    accuracy = model_eval(sess, x, y, predictions,
                          X_val, Y_val, args=eval_params)

    model_train(sess, x, y, predictions, X_train, Y_train,
                evaluate=model_eval(sess, x, y, predictions,
                                    X_val, Y_val,
                                    args=eval_params),
                args=train_params)

    print("Test accuracy on legitimate test examples: {}"
          .format(accuracy))

    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    adv_x = fgsm(x, predictions, eps=0.3)
    X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test],
                             args=eval_params)
    assert X_test_adv.shape[0] == 10000, X_test_adv.shape

    # Evaluate the accuracy of the MNIST model on adversarial examples
    adv1_accuracy = model_eval(sess, x, y, predictions, X_test_adv, Y_test,
                               args=eval_params)
    print("Test accuracy on adversarial examples: ".format(adv1_accuracy))

    print("Repeating the process, using adversarial training")
    # Redefine TF model graph
    model_2 = eval(args.model + '()')
    predictions_2 = model_2(x)
    adv_x_2 = fgsm(x, predictions_2, eps=0.3)
    predictions_2_adv = model_2(adv_x_2)

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
                evaluate=model_eval(sess, x, y, predictions_2_adv, X_test,
                                    Y_test, args=eval_params),
                args=train_params)
