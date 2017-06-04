from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import argparse
import keras.backend as K
from keras.utils.vis_utils import plot_model
from cleverhans.attacks import FastGradientMethod
# from cleverhans.utils_tf import model_argmax
# from cleverhans.utils import other_classes
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.utils import grid_visual, pair_visual

# from models import cnn_model
# from models import mlp_lle, cnn_lle, resnet
from utils import rank_classifiers, rank_features
from config import setup_config, setup_data
from attacks import evaluate_adversarial, whitebox_fgsm, jsma_attack
from attacks import prep_blackbox, substitute_model, train_sub
from utils import find_top_predictions, vis_cam
from sklearn.manifold import LocallyLinearEmbedding


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
                        help="Plot the network architecture.")
    parser.add_argument("-s", "--split_dataset", type=float, default=0.21,
                        help="Split the datasset. Keep % for validation.")
    parser.add_argument("-eps", "--epsilon", type=float, default=0.3,
                        help="Epsilon variable for adversarial distortion")
    parser.add_argument("-pv", "--pair_visual", type=int, default=5,
                        help="Plot normal and distorted image from "
                             "particular target classs")
    parser.add_argument("-gv", "--grid_visual", type=bool, default=False,
                        help="Plot a grid of all the images with distorted"
                             " in the diagonal")
    parser.add_argument("-ho", "--holdout", type=int, default=100,
                        help="Test set holdout for adversary.")
    parser.add_argument("-da", "--data_aug", type=int, default=False,
                        help="Wether to use augmentation during trainig"
                        " or not")
    parser.add_argument("-l", "--lmbda", type=float, default=0.2,
                        help="Lambda in https://arxiv.org/abs/1602.02697.")
    parser.add_argument("-nas", "--nb_attack_samples", type=int, default=10,
                        help="Nb of test set examples to attack")
    parser.add_argument("-pr", "--pretrained", type=bool, default=False,
                        help="Wether to load a pretrained model or not")
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    K.set_learning_phase(0)  # without this causes error during learning
    sess = setup_config()
    args = get_args()
    print("args = {}".format(args))
    trX, trY, valX, valY, teX, teY, x, y = setup_data(args)
    from utils import print_data_shapes

    if args.model == "mlp_lle":
        import tensorflow as tf
        trX = trX.reshape(-1, 784)
        teX = teX.reshape(-1, 784)
        valX = valX.reshape(-1, 784)
        x = tf.placeholder(tf.float32, shape=(None, 784))

    if args.model == "irnn":
        import tensorflow as tf
        trX = trX.reshape(-1, 784, 1)
        teX = teX.reshape(-1, 784, 1)
        valX = valX.reshape(-1, 784, 1)
        x = tf.placeholder(tf.float32, shape=(None, 784, 1))
        # for siamese
        # model, tr_pairs, tr_y, te_pairs, te_y = getattr(models,
        #                                                 args.model)(trX.shape[1:])
    if args.model == "variational_ae":
        trX = trX.reshape((len(trX), np.prod(trX.shape[1:])))
        teX = teX.reshape((len(teX), np.prod(teX.shape[1:])))

    print_data_shapes(trX, trY, valX, valY, teX, teY)

    # Define TF model graph
    if args.pretrained:
        from keras.models import load_model
        model = load_model('./models/' + args.model + '_'
                           + args.dataset + '.hdf5')
    else:
        import models
        if args.model == "variational_ae":
            model, encoder, generator = getattr(models,
                                                args.model)(trX.shape[1:])
            encoder.summary()
            generator.summary()
        else:
            model = getattr(models, args.model)(trX.shape[1:])

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
    if not args.pretrained and args.model == "variational_ae":
        from scipy.stats import norm
        import matplotlib.pyplot as plt
        model.fit(trX,
                  shuffle=True,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  validation_data=(teX, teX))
        # display a 2D plot of the digit classes in the latent space
        x_test_encoded = encoder.predict(teX, batch_size=args.batch_size)
        plt.figure(figsize=(6, 6))
        plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=teY)
        plt.colorbar()
        plt.show()

        # display a 2D manifold of the digits
        n = 15  # figure with 15x15 digits
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates on the unit square were transformed
        # Vthrough the inverse CDF (ppf) of the Gaussian
        # to produce values of the latent variables z, since the prior
        # of the latent space is Gaussian
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                x_decoded = generator.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit
                plt.figure(figsize=(10, 10))
                plt.imshow(figure, cmap='Greys_r')
                plt.show()
    else:
        model.fit(trX, trY, epochs=args.epochs,
                  batch_size=args.batch_size,
                  validation_data=(valX, valY), verbose=1)
        model.save('./models/' + args.model + '_' + args.dataset + '.hdf5')
    # # for siamese
    # model.fit(tr_pairs, trY, epochs=args.epochs, batch_size=args.batch_size,
    #           validation_data=(valX, valY), verbose=1)

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
        fgsm_par = {'eps': args.epsilon, 'ord': np.inf,
                    'clip_min': 0., 'clip_max': 1.}
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
        # Note: first pass X_test_adv through LLe and then evaluate
        # X_test_adv_lle = LocallyLinearEmbedding(n_neighbors=10,
        #                                         n_components=784,
        #                                         random_state=2017,
        #                                         n_jobs=-1).fit_transform(X_test_adv)
        adv_scores = model.evaluate(X_test_adv, teY)
        # print("Test accuracy on adversarial examples: ".format(accuracy))
        print("Test accuracy on adversarial examples: {}"
              .format(adv_scores[1]))

        # layer_name = 'conv2d_13'
        # preds = np.argmax(np.flip(np.sort(model.predict(teX), axis=0), axis=0),
        #                   axis=1)
        # imgs = trX[preds][:5]
        # vis_cam(model, imgs[0], layer_name)
        # find_top_predictions(model, args.model, layer_name, teX, teY,
        #                      X_test_adv, 5)
        # print("Repeating the process, using aversarial training")
        # # Redefine TF model graph
        # model_2 = eval(args.model + '()')
        # predictions_2 = model_2(x)
        # fgsm = FastGradientMethod(model_2, sess=sess)
        # adv_x_2 = fgsm.generate(x, **{'eps': args.epsilon})
        # predictions_2_adv = model_2(adv_x_2)

        # # Perform adversarial training
        # model_train(sess, x, y, predictions_2, trX, trY,
        #             predictions_adv=predictions_2_adv,
        #             args=train_params)

        # adv_acc = evaluate_adversarial(sess, x, y, predictions_2,
        #                                predictions_2_adv,
        #                                teX, teY, eval_params)

    if args.plot_arch is True:
        plot_model(model, to_file=eval(args.model + '.png'),
                   show_shapes=True,
                   show_layer_names=True)

    if args.rank_features is True:
        rank_features(np.vstack((trX, valX)).reshape(-1, 784),
                      np.argmax(np.vstack((trY, valY)), axis=1))
    if args.pair_visual is not None:
        pair_visual(teX[args.pair_visual].reshape(28, 28),
                    X_test_adv[args.pair_visual].reshape(28, 28))
        import pdb
        pdb.set_trace()

    if args.grid_visual is True:
        if args.dataset == "mnist":
            labels = np.unique(np.argmax(trY, axis=1))
            data = trX[labels]
        else:
            labels = np.unique(np.argmax(teY, axis=1))
            data = teX[labels]
        grid_visual(np.hstack((labels, data)))

    if args.rank_classifiers is True:
        from models import cnn_model, mlp, hierarchical, irnn, identity_model
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
