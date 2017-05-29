from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import argparse
import keras.backend as K
# from keras.utils.visualize_util import plot
from cleverhans.attacks import FastGradientMethod
# from cleverhans.utils_tf import model_argmax
# from cleverhans.utils import other_classes
from cleverhans.utils_tf import model_train, model_eval

from models import cnn_model
# from models import mlp_lle, cnn_lle, resnet
from utils import rank_classifiers, rank_features
from config import setup_config, setup_data
from attacks import evaluate_adversarial, whitebox_fgsm, jsma_attack
from attacks import prep_blackbox, substitute_model, train_sub
from utils import find_top_predictions


def get_args():
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
    parser.add_argument("-ds", "--dataset", type=str, required=True,
                        help="mnist | cifar10 | voc2012")
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
    parser.add_argument("-nas", "--nb_attack_samples", type=int, default=10,
                        help="Nb ot test set examples to attack")
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    K.set_learning_phase(0)  # without this causes error during learning
    sess = setup_config()
    args = get_args()
    print("args = {}".format(args))
    trX, trY, valX, valY, teX, teY, x, y = setup_data(args)

    # Define TF model graph
    model = eval(args.model + '()')
    model.summary()
    predictions = model(x)
    print("Defined TensorFlow graph.")
    train_params = {'nb_epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate,
                    'keras_learning_phase': 1}
    eval_params = {'batch_size': args.batch_size,
                   'keras_learning_phase': 0}

    # train on dataset
    model.fit(trX, trY, nb_epoch=args.epochs, batch_size=args.batch_size,
              validation_data=(valX, valY), verbose=1)

    # Evaluate accuracy of the MNIST model on legitimate test
    # examples
    scores = model.evaluate(teX, teY)
    print("Test accuracy on legitimate test examples: {}"
          .format(scores[1]))

    if args.attack == "jsma":
        jsma_attack(sess, model, x, y, predictions, args, teX, teY)
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
        # accuracy = model_eval(sess, x, y, preds_adv, X_test_adv, teY,
        #                       args=eval_params)
        adv_scores = model.evaluate(X_test_adv, teY)
        # print("Test accuracy on adversarial examples: ".format(accuracy))
        print("Test accuracy on adversarial examples: {}"
              .format(adv_scores[1]))

        from vis.visualization import visualize_cam
        layer_name = 'conv2d_2'
        layer_idx = layer_idx = [idx for idx, layer in enumerate(model.layers)
                                 if layer.name == layer_name][0]
        imgs = trX[np.argmax(trY, axis=1) == 5]
        img = np.expand_dims(imgs[0], axis=0)
        pred_class = np.argmax(model.predict(np.expand_dims(imgs[0], axis=0)))
        print("image shape {}, predicted_class = {}".format(img.shape,
                                                            pred_class))
        heatmap = visualize_cam(model, layer_idx, [pred_class], img)
        find_top_predictions(model, args.model, 'conv2d_2', teX, teY,
                             X_test_adv, 5)
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
