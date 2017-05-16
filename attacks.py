from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.datasets import cifar10, mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout

from cleverhans.utils import other_classes, cnn_model, pair_visual
from cleverhans.utils_tf import model_argmax
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import FastGradientMethod, SaliencyMapMethod
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation


def prep_blackbox(sess, x, y, args, trX, trY, teX, teY):
    """
    Define and train a model that simulates the "remote"
    black-box oracle described in the original paper.
    :param sess: the TF session
    :param x: the input placeholder for MNIST
    :param y: the ouput placeholder for MNIST
    :param trX: the training data for the oracle
    :param trY: the training labels for the oracle
    :param teX: the testing data for the oracle
    :param teY: the testing labels for the oracle
    :return:
    """

    # Define TF model graph (for the black-box model)
    model = cnn_model()
    predictions = model(x)
    print("Defined TensorFlow model graph.")

    # Train an MNIST model
    train_params = {
        'nb_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate
    }
    model_train(sess, x, y, predictions, trX, trY,
                verbose=False, args=train_params)

    # Print out the accuracy on legitimate data
    eval_params = {'batch_size': args.batch_size}
    accuracy = model_eval(sess, x, y, predictions, teX, teY,
                          args=eval_params)
    print('Test accuracy of black-box on legitimate test '
          'examples: ' + str(accuracy))

    return model, predictions


def substitute_model(img_rows=28, img_cols=28, classes=10):
    """
    Defines the model architecture to be used by the substitute
    :param img_rows: number of rows in input
    :param img_cols: number of columns in input
    :param classes: number of classes in output
    :return: keras model
    """
    model = Sequential()

    # Find out the input shape ordering
    if K.image_dim_ordering() == 'th':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)

    # Define a fully connected model (it's different than the black-box)
    layers = [Flatten(input_shape=input_shape),
              Dense(200),
              Activation('relu'),
              Dropout(0.5),
              Dense(200),
              Activation('relu'),
              Dropout(0.5),
              Dense(classes),
              Activation('softmax')]

    for layer in layers:
        model.add(layer)

    return model


def train_sub(sess, x, y, bbox_preds, X_sub, Y_sub, args):
    """
    This function creates the substitute by alternatively
    augmenting the training data and training the substitute.
    :param sess: TF session
    :param x: input TF placeholder
    :param y: output TF placeholder
    :param bbox_preds: output of black-box model predictions
    :param X_sub: initial substitute training data
    :param Y_sub: initial substitute training labels
    :return:
    """
    # Define TF model graph (for the black-box model)
    model_sub = substitute_model()
    preds_sub = model_sub(x)
    print("Defined TensorFlow model graph for the substitute.")

    # Define the Jacobian symbolically using TensorFlow
    grads = jacobian_graph(preds_sub, x, args.nb_classes)

    # Train the substitute and augment dataset alternatively
    for rho in xrange(args.data_aug):
        print("Substitute training epoch #" + str(rho))
        train_params = {
            # 'nb_epochs': args.nb_epochs_s,
            'nb_epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate
        }
        model_train(sess, x, y, preds_sub, X_sub, to_categorical(Y_sub),
                    verbose=False, args=train_params)

        # If we are not at last substitute training iteration, augment dataset
        if rho < args.data_aug - 1:
            print("Augmenting substitute training data.")
            # Perform the Jacobian augmentation
            X_sub = jacobian_augmentation(sess, x, X_sub, Y_sub, grads,
                                          args.lmbda)

            print("Labeling substitute training data.")
            # Label the newly generated synthetic points using the black-box
            Y_sub = np.hstack([Y_sub, Y_sub])
            X_sub_prev = X_sub[int(len(X_sub)/2):]
            eval_params = {'batch_size': args.batch_size}
            bbox_val = batch_eval(sess, [x], [bbox_preds], [X_sub_prev],
                                  args=eval_params)[0]
            # Note here that we take the argmax because the adversary
            # only has access to the label (not the probabilities) output
            # by the black-box model
            Y_sub[int(len(X_sub)/2):] = np.argmax(bbox_val, axis=1)

    return model_sub, preds_sub


def evaluate_adversarial(sess, x, y, predictions, predictions_adv,
                         X, Y, eval_params):
    # Evaluate the accuracy of the adversarialy trained MNIST model on
    # legitimate test examples
    adv2_accuracy = model_eval(sess, x, y, predictions, X, Y,
                               args=eval_params)
    print("Test accuracy on legitimate test examples: {}"
          .format(adv2_accuracy))

    # Evaluate the accuracy of the adversarially trained MNIST model on
    # adversarial examples
    adv3_accuracy = model_eval(sess, x, y, predictions_adv, X,
                               Y, args=eval_params)
    print("Test accuracy on adversarial examples: {}"
          .format(adv3_accuracy))


def whitebox_fgsm(sess, model, x, argues, teX, eval_params):
    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    fgsm = FastGradientMethod(model, sess=sess)
    fgsm_params = {'eps': argues.eps, 'keras_learning_phase': 0}
    adv_x = fgsm.generate(x, **fgsm_params)
    preds_adv = model(adv_x)
    teX_adv, = batch_eval(sess, [x], [adv_x], [teX],
                          args=eval_params)
    assert teX_adv.shape[0] == 10000, teX_adv.shape

    return adv_x, preds_adv, teX_adv


def jsma_attack(sess, model, x, y, preds, args, teX, teY):
    ###########################################################################
    # Craft adversarial examples using the Jacobian-based saliency map approach
    ###########################################################################
    print('Crafting ' + str(args.nb_attack_samples) + ' * ' +
          str(args.nb_classes-1) + ' adversarial examples')

    img_rows, img_cols, nb_channels = teX[0].shape

    # Keep track of success (adversarial example classified in target)
    results = np.zeros((args.nb_classes, args.nb_attack_samples), dtype='i')

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
            # jsma_params = {'theta': 1., 'gamma': 0.1,
            #                'nb_classes': args.nb_classes, 'clip_min': 0.,
            #                'clip_max': 1., 'targets': y,
            #                'y_val': one_hot_target}
            import pdb
            pdb.set_trace()
            adv_x = jsma.generate_np(teX[sample_ind:(sample_ind+1)],
                                     theta=1., gamma=0.1, nb_classes=10,
                                     clip_min=0., clip_max=1., targets=y,
                                     y_val=one_hot_target)

            # Check if success was achieved
            res = int(model_argmax(sess, x, preds, adv_x) == target)

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
