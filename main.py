from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import argparse
import os
# from keras.datasets import mnist
# from keras.utils.visualize_util import plot

import tensorflow as tf
from keras import backend as K

from cleverhans.attacks import FastGradientMethod, SaliencyMapMethod
from cleverhans.utils_tf import model_argmax
from cleverhans.utils import cnn_model, pair_visual, grid_visual
from cleverhans.utils import other_classes
from cleverhans.utils_tf import model_train, model_eval

from models import hierarchical, irnn, mlp, identity_model
from models import mlp_lle, cnn_lle
from utils import rank_classifiers, rank_features
from attacks import setup_config, setup_data
from attacks import evaluate_adversarial, whitebox_fgsm, jsma_attack
from attacks import prep_blackbox, substitute_model, train_sub


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
                        "1.)fgsm, 2.)jsma")
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
    parser.add_argument("-ho", "--holdout", type=int, default=100,
                        help="Test set holdout for adversary.")
    parser.add_argument("-da", "--data_aug", type=int, default=6,
                        help="Training epochs for each substitute.")
    parser.add_argument("-l", "--lmbda", type=float, default=0.2,
                        help="Lambda in https://arxiv.org/abs/1602.02697.")
    parser.add_argument("-td", "--train_dir", type=str, default="./tmp",
                        help="Directory storing the saved model.")
    parser.add_argument("-f", "--filename", type=str, default="mnist.ckpt",
                        help="Filename to save model under.")
    parser.add_argument("-nas", "--nb_attack_samples", type=int, default=10,
                        help="Nb ot test set examples to attack")
    args = parser.parse_args()

    K.set_learning_phase(0)  # without this causes error during learning
    sess = setup_config()
    # tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
    trX, trY, valX, valY, teX, teY, x, y = setup_data(args)

    # Define TF model graph
    model = eval(args.model + '()')
    model.summary()
    predictions = model(x)
    print("Defined TensorFlow graph.")
    # tf.global_variables_initializer()
    train_params = {'nb_epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate}
    eval_params = {'batch_size': args.batch_size}

    # Train an MNIST model if it does not exist in the train_dir folder
    # saver = tf.train.Saver()
    # save_path = os.path.join(args.train_dir, args.filename)
    # if os.path.isfile(save_path):
    #     saver.restore(sess, os.path.join(args.train_dir, args.filename))
    # else:
    # train on dataset
    model_train(sess, x, y, predictions, trX, trY, args=train_params)
    # saver.save(sess, save_path)

    # Evaluate the accuracy of the MNIST model on legitimate validation
    # examples
    accuracy = model_eval(sess, x, y, predictions, valX, valY,
                          args=eval_params)
    print("Test accuracy on legitimate test examples: {}"
          .format(accuracy))

    if args.attack == "jsma":
        # jsma_attack(sess, model, x, y, predictions, args, teX, teY)
        ###########################################################################
        # Craft adversarial examples using the Jacobian-based saliency map approach
        ###########################################################################
        print('Crafting ' + str(args.nb_attack_samples) + ' * ' +
              str(args.nb_classes-1) + ' adversarial examples')

        img_rows, img_cols, nb_channels = teX[0].shape

        # Keep track of success (adversarial example classified in target)
        results = np.zeros((args.nb_classes, args.nb_attack_samples),
                           dtype='i')

        # Rate of perturbed features for each test set example and target class
        perturbations = np.zeros((args.nb_classes, args.nb_attack_samples),
                                 dtype='f')

        # Initialize our array for grid visualization
        grid_shape = (args.nb_classes,
                      args.nb_classes,
                      img_rows,
                      img_cols,
                      nb_channels)
        grid_viz_data = np.zeros(grid_shape, dtype='f')

        # Define the SaliencyMapMethod attack object
        jsma = SaliencyMapMethod(model, back='tf', sess=sess)

        # Loop over the samples we want to perturb into adversarial examples
        for sample_ind in xrange(0, args.nb_attack_samples):
            print('--------------------------------------')
            print("Attacking input {}/{}".format(sample_ind + 1,
                                                 args.nb_attack_samples))

            # We want to find an adversarial example for each possible target class
            # (i.e. all classes that differ from the label given in the dataset)
            current_class = int(np.argmax(teY[sample_ind]))
            target_classes = other_classes(args.nb_classes, current_class)

            # For the grid visualization, keep original images along the diagonal
            grid_viz_data[current_class, current_class, :, :, :] = np.reshape(
                teX[sample_ind:(sample_ind+1)],
                (img_rows, img_cols, nb_channels))

            # Loop over all target classes
            for target in target_classes:
                print('Generating adv. example for target class %i' % target)

                # This call runs the Jacobian-based saliency map approach
                one_hot_target = np.zeros((1, args.nb_classes), dtype=np.float32)
                one_hot_target[0, target] = 1
                jsma_params = {'theta': 1., 'gamma': 0.1,
                               'nb_classes': args.nb_classes, 'clip_min': 0.,
                               'clip_max': 1., 'targets': y,
                               'y_val': one_hot_target}
                import pdb
                pdb.set_trace()
                adv_x = jsma.generate_np(teX[sample_ind:(sample_ind+1)],
                                         **jsma_params)

                # Check if success was achieved
                res = int(model_argmax(sess, x, predictions, adv_x) == target)

                # Compute number of modified features
                adv_x_reshape = adv_x.reshape(-1)
                test_in_reshape = teX[sample_ind].reshape(-1)
                nb_changed = np.where(adv_x_reshape != test_in_reshape)[0].shape[0]
                percent_perturb = float(nb_changed) / adv_x.reshape(-1).shape[0]

                # Display the original and adversarial images side-by-side
                if args.grid_visual:
                    if 'figure' not in vars():
                        figure = pair_visual(
                            np.reshape(teX[sample_ind:(sample_ind+1)],
                                       (img_rows, img_cols)),
                            np.reshape(adv_x,
                                       (img_rows, img_cols)))
                    else:
                        figure = pair_visual(
                            np.reshape(teX[sample_ind:(sample_ind+1)],
                                       (img_rows, img_cols)),
                            np.reshape(adv_x, (img_rows,
                                       img_cols)), figure)

                # Add our adversarial example to our grid data
                grid_viz_data[target, current_class, :, :, :] = np.reshape(
                    adv_x, (img_rows, img_cols, nb_channels))

                # Update the arrays for later analysis
                results[target, sample_ind] = res
                perturbations[target, sample_ind] = percent_perturb

        print('--------------------------------------')

        # Compute the number of adversarial examples that were successfully found
        nb_targets_tried = ((args.nb_classes - 1) * args.nb_attack_samples)
        succ_rate = float(np.sum(results)) / nb_targets_tried
        print('Avg. rate of successful adv. examples {0:.4f}'.format(succ_rate))

        # Compute the average distortion introduced by the algorithm
        percent_perturbed = np.mean(perturbations)
        print('Avg. rate of perturbed features {0:.4f}'.format(percent_perturbed))

        # Compute the average distortion introduced for successful samples only
        percent_perturb_succ = np.mean(perturbations * (results == 1))
        print('Avg. rate of perturbed features for successful '
              'adversarial examples {0:.4f}'.format(percent_perturb_succ))
    elif args.attack == "blackbox":
        # Initialize substitute training set reserved for adversary
        subX = teX[:args.holdout]
        subY = np.argmax(teY[:args.holdout], axis=1)

        # Redefine test set as remaining samples unavailable to adversaries
        teX = teX[args.holdout:]
        teY = teY[args.holdout:]

        # Simulate the black-box model locally
        # You could replace this by a remote labeling API for instance
        print("Preparing the black-box model.")
        model, bbox_preds = prep_blackbox(sess, x, y, args, trX, trY, teX, teY)

        print("Training the substitute model.")
        # Train substitute using method from https://arxiv.org/abs/1602.02697
        model_sub, preds_sub = train_sub(sess, x, y, bbox_preds, subX, subY,
                                         args)

        # Initialize the Fast Gradient Sign Method (FGSM) attack object.
        fgsm_par = {'eps': 0.3, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.}
        fgsm = FastGradientMethod(model_sub, sess=sess)

        # Craft adversarial examples using the substitute
        x_adv_sub = fgsm.generate(x, **fgsm_par)

        # Evaluate the accuracy of the "black-box" model on adversarial
        # examples
        accuracy = model_eval(sess, x, y, model(x_adv_sub), teX, teY,
                              args=eval_params)
        print('Test accuracy of oracle on adversarial examples generated '
              'using the substitute: ' + str(accuracy))
    elif args.attack == "fgsm":
        # craft adversarial examples using fgsm
        adv_x, preds_adv, X_test_adv = whitebox_fgsm(sess, model, x, args,
                                                     teX, eval_params)

        # Evaluate the accuracy of the MNIST model on adversarial examples
        # X_test_adv might change to X_test as in the new cleverhans example
        accuracy = model_eval(sess, x, y, preds_adv, X_test_adv, teY,
                              args=eval_params)
        print("Test accuracy on adversarial examples: ".format(accuracy))

        print("Repeating the process, using aversarial training")
        # Redefine TF model graph
        model_2 = eval(args.model + '()')
        predictions_2 = model_2(x)
        fgsm = FastGradientMethod(model_2, sess=sess)
        adv_x_2 = fgsm.generate(x, **{'eps': args.eps})
        predictions_2_adv = model_2(adv_x_2)

        # Perform adversarial training
        model_train(sess, x, y, predictions_2, trX, trY,
                    predictions_adv=predictions_2_adv,
                    args=train_params)

        adv_acc = evaluate_adversarial(sess, x, y, predictions_2,
                                       predictions_2_adv,
                                       teX, teY, eval_params)

    if args.plot_arch is True:
        plot(model, to_file=eval(args.model + '.png'), show_shapes=True,
             show_layer_names=True)

    if args.rank_features is True:
        rank_features(np.vstack((trX, valX)).reshape(-1, 784),
                      np.argmax(np.vstack((trY, valY)), axis=1))
    # if args.pair_visual is not None:
    #     pair_visual(teX[args.pair_visual].reshape(28, 28),
    #                 X_test_adv[args.pair_visual].reshape(28, 28))

    if args.grid_visual is True:
        if args.dataset == "mnist":
            labels = np.unique(np.argmax(trY, axis=1))
            data = trX[labels]
        else:
            labels = np.unique(np.argmax(teY, axis=1))
            data = teX[labels]
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

        rank_classifiers(models, np.vstack((trX, valX)),
                         np.vstack((trY, valY)),
                         teX, X_test_adv,
                         teY, epochs=args.epochs,
                         batch_size=args.batch_size)

    # Close TF session
    sess.close()
