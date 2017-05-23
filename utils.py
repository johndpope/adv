import numpy as np
from keras import backend as K
from keras.layers import Lambda
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import normalize
from deap import algorithms, tools
from scipy.spatial import distance
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim
import cv2
# import multiprocessing as mp
import itertools
import scipy
from operator import attrgetter
from cleverhans.utils import pair_visual
from grad_cam import run_gradcam


def roc_auc(teY, teY_pred, counter, color, mean_tpr, mean_fpr):
    # Compute ROC curve and area under curve, main k-fold loop
    fpr, tpr, thresholds = roc_curve(np.argmax(teY, axis=1),
                                     np.max(teY_pred, axis=1),
                                     pos_label=counter)
    mean_tpr += scipy.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, color=color,
             label='ROC fold {} (area = {:0.2f})'
             .format(counter, roc_auc))
    plt.hold(True)

    return mean_tpr, mean_fpr


def plot_roc_auc(X, Y, skf, mean_tpr, mean_fpr):
    mean_tpr /= skf.get_n_splits(X, Y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
             label='Random')
    plt.hold(True)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='Mean ROC (area = {:0.2f})'.format(mean_auc, lw=2))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()


def rank_classifiers(models, X, Y, X_test, X_test_adv, Y_test,
                     save_model=False, epochs=2, batch_size=128):
    """
    models: list of tuples [('model name', model object), ...]
    X: training data
    Y: labels, should be one-hot and not clipped
    """
    colors = itertools.cycle(['cyan', 'indigo', 'seagreen', 'yellow',
                              'blue', 'darkorange'])
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    cv_results = []
    results = []
    names = []
    counter = 0
    skf = StratifiedKFold(n_splits=10, random_state=2017, shuffle=True)
    fig = plt.figure()
    ax_roc = fig.add_subplot(111)
    for name, model in models:
        print("\n\nTesting model {}\n".format(name, counter))
        model.summary()
        for (tr_id, te_id), color in zip(skf.split(X, np.argmax(Y, axis=1)),
                                         colors):
            print("Fold {}".format(counter))
            trX, teX = X[tr_id], X[te_id]
            trY, teY = Y[tr_id], Y[te_id]
            if name == "mlp":
                trX = trX.reshape(-1, 784)
                teX = teX.reshape(-1, 784)
                X_test = X_test.reshape(-1, 784)
                X_test_adv = X_test_adv.reshape(-1, 784)
            elif name == "irnn":
                trX = trX.reshape(-1, 784, 1)
                teX = teX.reshape(-1, 784, 1)
                X_test = X_test.reshape(-1, 784, 1)
                X_test_adv = X_test_adv.reshape(-1, 784, 1)
            else:
                X_test = X_test.reshape(-1, 28, 28, 1)
                X_test_adv = X_test_adv.reshape(-1, 28, 28, 1)
            model.fit(trX, trY, nb_epoch=epochs, batch_size=batch_size,
                      validation_split=0.2, verbose=1)
            scores = model.evaluate(teX, teY, verbose=0)
            teY_pred = model.predict(teX)
            cv_results.append(scores[1])
            mean_tpr, mean_fpr = roc_auc(teY, teY_pred, counter,
                                         color, mean_tpr, mean_fpr)
            counter += 1
        counter = 0
        legit_scores = model.evaluate(X_test, Y_test)
        adv_scores = model.evaluate(X_test_adv, Y_test)
        print("\nmodel = {}, mean = {}, std = {}, legit test acc. = {}, "
              "adv. test acc. = {}"
              .format(name, np.mean(cv_results), np.std(cv_results),
                      legit_scores[1], adv_scores[1]))
        # results.append([np.mean(cv_results), np.std(cv_results)])
        results.append(cv_results)
        cv_results = []
        names.append(name)
        teY_pred = model.predict(teX)
        report = classification_report(np.argmax(teY, axis=1),
                                       np.argmax(teY_pred, axis=1))
        print(report)
        plot_roc_auc(X, Y, skf, mean_tpr, mean_fpr)
        model.save_model("{}.hdf5".format(name))
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    plt.ylabel('Accuracy')
    ax.set_xticklabels(names)
    plt.show()


def plot_2d_embedding(X, y, X_embedded, name, min_dist=10.0):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(frameon=False)
    plt.title("\\textbf{MNIST dataset} -- Two-dimensional "
              "embedding of %s" % name)
    plt.setp(ax, xticks=(), yticks=())
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                        wspace=0.0, hspace=0.0)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, marker="x")

    if min_dist is not None:
        from matplotlib import offsetbox
        shown_images = np.array([[15., 15.]])
        indices = np.arange(X_embedded.shape[0])
        np.random.shuffle(indices)
        for i in indices[:5000]:
            dist = np.sum((X_embedded[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist:
                continue
            shown_images = np.r_[shown_images, [X_embedded[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(X[i].reshape(28, 28),
                                      cmap=plt.cm.gray_r),
                X_embedded[i])
            ax.add_artist(imagebox)


def ga_plot_results(filename, gen, fitness_maxs, fitness_avgs):
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fitness_maxs, "r-", label="Maximum Fitness")
    line2 = ax1.plot(gen, fitness_avgs, "b-", label="Average Fitness")
    lines = line1 + line2
    labs = [line.get_label() for line in lines]
    ax1.legend(lines, labs, loc="lower right")
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    plt.savefig('{}'.format(filename))


def global_average_pooling(x):
    return K.mean(x, axis=(2, 3))


def global_average_pooling_shape(input_shape):
    return input_shape[0:2]


def global_avg_pooling_layer(model):
    model.add(Lambda(global_average_pooling,
                     output_shape=global_average_pooling_shape))

    return model


def get_classmap(model, X, nb_classes, batch_size, num_input_channels, ratio):
    """Notice: it requires theano as backend"""
    from theano.tensor import absconv

    inc = model.layers[0].input
    conv6 = model.layers[-4].output  # this corresponds to the last
    # conv layer's output from vgg16
    # new_height = int(round(old_height * scale)) this could replace
    # new_width = int(round(old_width * scale))   absconv from theano
    # resized = tf.image.resize_images(input_tensor, [new_height, new_width])
    # resized = tf.image.resize_bilinear(input_tensor, [new_height, new_width],
    #                                    align_corners=None)
    conv6_resized = absconv.bilinear_upsampling(conv6,
                                                ratio,
                                                batch_size=batch_size,
                                                num_input_channels=num_input_channels)
    WT = model.layers[-1].W.T  # this corresponds to the softmax layer
    # of vgg16. The transpose operator doesn't work with tf.
    conv6_resized = K.reshape(conv6_resized,
                              (-1, num_input_channels, 224 * 224))
    classmap = K.dot(WT, conv6_resized).reshape((-1, nb_classes, 224, 224))
    get_cmap = K.function([inc], classmap)
    return get_cmap([X])


def plot_classmap(model, img_path, label,
                  nb_classes=10, num_input_channels=1024, ratio=16):
    """
    Plot class activation map of trained VGGCAM model
    args: VGGCAM_weight_path (str) path to trained keras VGGCAM weights
          img_path (str) path to the image for which we get the activation map
          label (int) label (0 to nb_classes-1) of the class activation map to plot
          nb_classes (int) number of classes
          num_input_channels (int) number of conv filters to add
                                   in before the GAP layer
          ratio (int) upsampling ratio (16 * 14 = 224)
    """

    # Load, add number of classes and input channels and compile model
    # model = VGGCAM(nb_classes, num_input_channels)
    # model.load_weights(VGGCAM_weight_path)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")

    # Load and format data
    im = cv2.resize(cv2.imread(img_path), (224, 224)).astype(np.float32)
    # Get a copy of the original image
    im_ori = im.copy().astype(np.uint8)
    # VGG model normalisations
    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68
    im = im.transpose((2, 0, 1))

    batch_size = 1
    classmap = get_classmap(model,
                            im.reshape(1, 3, 224, 224),
                            nb_classes,
                            batch_size,
                            num_input_channels=num_input_channels,
                            ratio=ratio)

    plt.imshow(im_ori)
    plt.imshow(classmap[0, label, :, :],
               cmap="jet",
               alpha=0.5,
               interpolation='nearest')
    plt.show()
    raw_input()


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


def visualize_class_activation_map(model, img_path, output_path,
                                   layer_name="conv5_3", target_class=1):
        # model = load_model(model_path)
        original_img = cv2.imread(img_path, 1)
        width, height, _ = original_img.shape

        # Reshape to the network input shape (3, w, h).
        img = np.array([np.transpose(np.float32(original_img), (2, 0, 1))])

        # Get the 512 input weights to the softmax.
        class_weights = model.layers[-1].get_weights()[0]
        final_conv_layer = model.get_output_layer(model, layer_name)
        get_output = K.function([model.layers[0].input],
                                [final_conv_layer.output,
                                 model.layers[-1].output])
        [conv_outputs, predictions] = get_output([img])
        conv_outputs = conv_outputs[0, :, :, :]

        # Create the class activation map.
        cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[1:3])
        for i, w in enumerate(class_weights[:, target_class]):
                cam += w * conv_outputs[i, :, :]
        print("predictions", predictions)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (height, width))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam < 0.2)] = 0
        img = heatmap * 0.5 + original_img
        cv2.imwrite(output_path, img)


def rank_features(X, y):
    if X.ndim > 2 or y.ndim == 2:
        raise ValueError("X should be 2d array while "
                         "y should be 1d label array.")
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        random_state=2017)
    # X_train = X_train[:, :10]
    # X_test = X_test[:, :10]
    model = XGBClassifier()
    model.fit(X, y)
    print(model.feature_importances_[
        np.where(model.feature_importances_ > 0)])
    plt.figure(figsize=(50, 20))
    plot_importance(model)
    plt.show()
    # make predictions for test data and evaluate
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: {}%%".format(accuracy * 100.0))
    # Fit model using each importance as a threshold
    thresholds = np.sort(model.feature_importances_[
        np.where(model.feature_importances_ > 0)])
    # thresholds = np.sort(model.feature_importances_)
    for thresh in thresholds:
        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
        # train model
        selection_model = XGBClassifier()
        selection_model.fit(select_X_train, y_train)
        # eval model
        select_X_test = selection.transform(X_test)
        y_pred = selection_model.predict(select_X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        print("Thresh={}, n={}, Accuracy: {}%%".format(thresh,
                                                       select_X_train.shape[1],
                                                       accuracy * 100.0))


def flatten_parameters(model):
    """
    Gets the weights from the network layers and puts them in one flat
    parameter vector
    """
    return np.concatenate([layer.flatten() for layer in model.get_weights()])


def update_model_weights(model, new_weights):
    """
    Updates the network with new weights after they have been stored in one
    flat parameter vector
    """
    accum = 0
    for layer in model.layers:
        current_layer_weights_list = layer.get_weights()
        new_layer_weights_list = []
        for layer_weights in current_layer_weights_list:
            layer_total = np.prod(layer_weights.shape)
            new_layer_weights_list.append(
                new_weights[accum:accum + layer_total]
                .reshape(layer_weights.shape))
            accum += layer_total
        layer.set_weights(new_layer_weights_list)


# This is a modified version of the eaSimple algorithm included with DEAP here:
# https://github.com/DEAP/deap/blob/master/deap/algorithms.py#L84
def eaSimpleModified(population, toolbox, cxpb, mutpb, ngen, stats=None,
                     halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    best = []

    best_ind = max(population, key=attrgetter("fitness"))
    best.append(best_ind)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Save the best individual from the generation
        best_ind = max(offspring, key=attrgetter("fitness"))
        best.append(best_ind)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook, best


def ga_run(toolbox, num_gen=10, n=100, mutpb=0.8, cxpb=0.5):
    """
    Runs multiple episodes, evolving the model parameters using a GA
    """
    np.random.seed(2017)
    history = tools.History()
    # Decorate the variation operators
    toolbox.decorate("mate", history.decorator)
    toolbox.decorate("mutate", history.decorator)

    # pool = mp.Pool(processes=12)
    # toolbox.register("map", pool.map)

    pop = toolbox.population(n=n)
    history.update(pop)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # pop, log = algorithms.eaSimple(pop,
    #                                toolbox,
    #                                cxpb=cxpb,
    #                                mutpb=mutpb,
    #                                ngen=num_gen,
    #                                stats=stats,
    #                                halloffame=hof,
    #                                verbose=True)
    pop, log = algorithms.eaSimpleModified(pop,
                                           toolbox,
                                           cxpb=cxpb,
                                           mutpb=mutpb,
                                           ngen=num_gen,
                                           stats=stats,
                                           halloffame=hof,
                                           verbose=True)

    return pop, log, hof, history


def ga_train(model, data, output_dir='./tmp'):
    try:
        NUM_GENERATIONS = 100
        POPULATION_SIZE = 96
        # MUTATION_PROB = 0.02
        CROSSOVER_PROB = 0.5

        MUTATION_PROBS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

        for mutation_prob in MUTATION_PROBS:
            pop, log, hof, history, best_per_gen = ga_run(
                num_gen=NUM_GENERATIONS,
                n=POPULATION_SIZE,
                cxpb=CROSSOVER_PROB,
                mutpb=mutation_prob
            )

        best = np.asarray(hof)
        gen = log.select("gen")
        fitness_maxs = log.select("max")
        fitness_avgs = log.select("avg")

        # try this
        ga_plot_results(filename='{}train_cnn_ga_mutpb_{}.png'
                        .format(output_dir,
                                str(mutation_prob).replace('.', '_')),
                        gen=gen,
                        fitness_maxs=fitness_maxs,
                        fitness_avgs=fitness_avgs)

        np.savetxt('{}train_cnn_ga_mutpb_{}.out'.
                   format(output_dir,
                          str(mutation_prob).replace('.', '_')), best)

        # Plot the feature vectors produced by the best individual from each
        # generation
        for gen in xrange(len(best_per_gen)):
            update_model_weights(model, np.asarray(best_per_gen[gen]))
            feature_vectors = calculate_model_output(model, data)
            # plot_feature_vectors(feature_vectors,
            #                      filename='{}feature_vectors_{}__{}.png'
            #                      .format(output_dir,
            #                              str(mutation_prob)
            #                              .replace('.', '_'),
            #                              gen))


        # or that
        # Plot the results
        plt.plot(fitness_maxs)  # , '.')
        plt.plot(fitness_avgs)  # , '.')
        plt.legend(['maximum', 'average'], loc=4)
        plt.xlabel('Episode')
        plt.ylabel('Fitness')

        # Save the results to disk
        np.savetxt('weights.out', best)
        np.savetxt('fitness_avgs.out', fitness_avgs)
        np.savetxt('fitness_maxs.out', fitness_maxs)

        individuals = []
        for i in history.genealogy_history.items():
            individuals.append(i[1])
        inp = np.array(individuals)
        np.savetxt('history.out', inp)

        plt.savefig('learning_history.png')
        plt.show()

    finally:
        pass


def calculate_model_output(model, input, multiple=False):
    output = model.predict(input)
    if multiple:
        output = output.reshape(output.shape[0], output.shape[1])
    else:
        output = output.reshape(output.shape[1])

    normalized_output = normalize(output)

    return normalized_output


def calculate_fitness(feature_vectors):
    pairwise_euclidean_distances = distance.pdist(feature_vectors, 'euclidean')
    fitness = pairwise_euclidean_distances.mean()
    + pairwise_euclidean_distances.min()

    return fitness


def find_top_predictions(model, model_name, layer_name, teX, teY,
                         teX_adv, count):
    preds = model.predict(teX)
    preds_adv = model.predict(teX_adv)
    rows, cols = np.where(preds >= np.max(preds, axis=0))
    targets = np.argmax(preds[rows], axis=1)[:count]
    orig_targets = np.argmax(teY[rows], axis=1)[:count]
    accuracies = preds[rows]
    accuracies = accuracies[:count]
    accuracies_adv = preds_adv[rows]
    adv_accuracies = accuracies_adv[:count]
    imgs = teX[targets]
    imgs = imgs[:count]
    print("top image labels are {}".format(targets))
    print("accuracies for top image labels are {}".format(
        np.max(accuracies, axis=1)))
    print("accuracies for top adv. image labels are {}".format(
        np.max(adv_accuracies, axis=1)))
    print("true labels for top images are {}".format(orig_targets))
    print("top images are {}".format(imgs.shape))
    imgs_adv = teX_adv[rows]
    imgs_adv = imgs_adv[:count]
    print("top adv. images are {}".format(imgs_adv.shape))
    for key, val in enumerate(imgs_adv):
        pair_visual(imgs[key].reshape(28, 28),
                    imgs_adv[key].reshape(28, 28))
        run_gradcam(model, model_name, imgs[key], orig_targets[key],
                    layer_name)


def mse(imgA, imgB):
    if imgA.ndim == 3 and imgB.ndim == 3:
        if imgA.shape[2] == 3 and imgB.shape[2] == 3:
            err = 0.
            for c in xrange(imgA.shape[2]):
                err += np.sum((imgA[:, :, c].astype('float32')
                               - imgB[:, :, c].astype('float32')) ** 2)
            err /= np.prod(imgA.shape[:3])  # maybe also x3
        else:
            if imgA.ndim == 3 and imgB.ndim == 3:
                if imgA.shape[2] == 1 and imgB.shape[2] == 1:
                    imgA = imgA.reshape(imgA.shape[0], imgA.shape[1])
                    imgB = imgB.reshape(imgB.shape[0], imgB.shape[1])

            err = np.sum((imgA.astype('float32') -
                          imgB.astype('float32')) ** 2)
            err /= float(np.prod(imgA.shape[:2]))

    psnr = 20.0 * np.log10(np.max(imgA)) - 10.0 * np.log10(err)  # np.max(imgA)
    # should be 255.

    return psnr, err


def ssim(imgA, imgB):
    return compare_ssim(imgA, imgB, multichannel=True)
