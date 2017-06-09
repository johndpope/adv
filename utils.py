# coding: utf-8

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
import networkx
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


def estimator_hyperparam_sensitivity(estimator, X, y, hyperparam="alpha"):
    """
    If the training score and the validation score are both low, the
    estimator will be underfitting.
    If the training score is high and the validation score is low, the
    estimator is overfitting and otherwise
    it is working very well.
    A low training score and a high validation
    score is usually not possible.
    """
    import numpy as np
    from sklearn.model_selection import validation_curve, learning_curve
    # from sklearn.datasets import load_iris
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVC

    np.random.seed(2017)
    # iris = load_iris()
    # X, y = iris.data, iris.target
    # indices = np.arange(y.shape[0])
    # np.random.shuffle(indices)
    # X, y = X[indices], y[indices]
    train_scores, valid_scores = validation_curve(Ridge(), X, y, "alpha",
                                                  np.logspace(-7, 3, 3))

    train_sizes, train_scores, valid_scores = learning_curve(
        SVC(kernel='linear'), X, y, train_sizes=[50, 80, 110],
        cv=5)


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
    model.add(Lambda(
        global_average_pooling,
        output_shape=global_average_pooling_shape
        )
    )
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


def visualize_class_activation_map(model, img, layer_name="conv5_3",
                                   target_class=1,
                                   output_path='./gradcam/gradimg.png'):
    # model = load_model(model_path)
    # original_img = cv2.imread(img_path, 1)
    width, height, _ = img.shape

    # Reshape to the network input shape (3, w, h).
    # img = np.array([np.transpose(np.float32(original_img), (2, 0, 1))])

    # Get the 512 input weights to the softmax.
    class_weights = model.layers[-1].get_weights()[0]
    final_conv_layer = get_output_layer(model, layer_name)
    get_output = K.function([model.layers[0].input],
                            [final_conv_layer.output,
                             model.layers[-1].output])
    [conv_outputs, predictions] = get_output([np.expand_dims(img, axis=0)])
    conv_outputs = conv_outputs[0, :, :, :]

    # Create the class activation map.
    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[:2])
    counter = 0
    for w in class_weights[:, target_class]:
        if counter > conv_outputs.shape[2] - 1:
            counter = 0
        cam += w * conv_outputs[:, :, counter]
        counter += 1
    print("predictions", predictions)
    import pdb; pdb.set_trace() ## DEBUG ##
    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    img = heatmap * 0.5 + img
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
    plt.figure(figsize=(50, 50))
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
        # to create a copy of the selected individuals or clone
        # offspring = [toolbox.clone(ind) for ind in offspring]
        # offspring = map(toolbox.clone, offspring)
        # select and clone individuals in one line
        # offspring = map(toolbox.clone, toolbox.select(population,
        #                                               len(population)))

    #     # Apply crossover and mutation on the offspring
    #     for child1, child2 in zip(offspring[::2], offspring[1::2]):
    #         if np.random.random() < cxpb:
    #             toolbox.mate(child1, child2)
    #             del child1.fitness.values
    #             del child2.fitness.values

    #     for mutant in offspring:
    #         if np.random.random() < mutpb:
    #             toolbox.mutate(mutant)
    #             del mutant.fitness.values

        # Vary the pool of individuals/mutate and crossover
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


def ga_plot_genealogy(history, hof, toolbox, max_depth=5):
    h = history.getGenealogy(hof[0], max_depth=max_depth)
    graph = networkx.DiGraph(h)
    graph = graph.reverse()
    colors = [toolbox.evaluate(history.genealogy_history[i])[0] for i in graph]
    pos = networkx.graphviz_layout(graph, prog="dot")
    networkx.draw(graph, pos, node_color=colors)
    cb = plt.colorbat()
    cb.set_label("Error")
    plt.show()


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

    hof = tools.HallOfFame(maxsize=1)
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
    pop, log, best = eaSimpleModified(pop,
                                      toolbox,
                                      cxpb=cxpb,
                                      mutpb=mutpb,
                                      ngen=num_gen,
                                      stats=stats,
                                      halloffame=hof,
                                      verbose=True)

    return pop, log, hof, history, best


def ga_train(model, data, toolbox, genealogy=False, output_dir='./tmp'):
    NUM_GENERATIONS = 100
    POPULATION_SIZE = 96
    # MUTATION_PROB = 0.02
    CROSSOVER_PROB = 0.5

    MUTATION_PROBS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

    for mutation_prob in MUTATION_PROBS:
        pop, log, hof, history, best_per_gen = ga_run(
            toolbox,
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

    # # or that
    # # Plot the results
    # plt.plot(fitness_maxs)  # , '.')
    # plt.plot(fitness_avgs)  # , '.')
    # plt.legend(['maximum', 'average'], loc=4)
    # plt.xlabel('Episode')
    # plt.ylabel('Fitness')

    # # Save the results to disk
    # np.savetxt('weights.out', best)
    # np.savetxt('fitness_avgs.out', fitness_avgs)
    # np.savetxt('fitness_maxs.out', fitness_maxs)

    # individuals = []
    # for i in history.genealogy_history.items():
    #     individuals.append(i[1])
    # inp = np.array(individuals)
    # np.savetxt('history.out', inp)

    # plt.savefig('learning_history.png')
    # plt.show()

    if genealogy:
        ga_plot_genealogy(history, hof, toolbox)


def calculate_model_output(model, input, multiple=False):
    output = model.predict(input)
    if multiple:
        output = output.reshape(-1, np.prod(output.shape[1:]))
    else:
        # output = output.reshape(output.shape[1])
        output = output

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


def ga_checkpoint_deap(checkpoint=None):
    if checkpoint:
        # A file name has been given, then load the data from the file
        with open(checkpoint, "r") as cp_file:
            cp = pickle.load(cp_file)
        population = cp["population"]
        start_gen = cp["generation"]
        halloffame = cp["halloffame"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
    else:
        # Start a new evolution
        population = toolbox.population(n=300)
        start_gen = 0
        halloffame = tools.HallOfFame(maxsize=1)
        logbook = tools.Logbook()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("max", numpy.max)

    for gen in range(start_gen, NGEN):
        population = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        halloffame.update(population)
        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)

        population = toolbox.select(population, k=len(population))

        if gen % FREQ == 0:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=population, generation=gen, halloffame=halloffame,
                      logbook=logbook, rndstate=random.getstate())

            with open("checkpoint_name.pkl", "wb") as cp_file:
                pickle.dump(cp, cp_file)


def mse(imgA, imgB):
    err = 0.
    if imgA.ndim == 3 and imgB.ndim == 3:
        if imgA.shape[2] == 3 and imgB.shape[2] == 3:
            for c in xrange(imgA.shape[2]):
                err += np.sum((imgA[:, :, c].astype('float32')
                               - imgB[:, :, c].astype('float32')) ** 2)
            err /= (np.prod(imgA.shape[:2]) + 1e-5)  # maybe also x3
    # else:
    #     if imgA.ndim == 3 and imgB.ndim == 3:
        if imgA.shape[2] == 1 and imgB.shape[2] == 1:
            imgA = imgA.reshape(imgA.shape[0], imgA.shape[1])
            imgB = imgB.reshape(imgB.shape[0], imgB.shape[1])

    err = np.sum((np.float32(imgA) - np.float32(imgB)) ** 2)
    err = err / (np.float32(np.prod(imgA.shape)) + 1e-5)

    psnr = 20.0 * np.log10(np.max(imgA)) - 10.0 * np.log10(err)  # np.max(imgA)
    # should be 255.

    return psnr, err


def ssim(imgA, imgB):
    return compare_ssim(np.float32(imgA), np.float32(imgB), multichannel=True)


def plot_feature_ranking(X, ranking):
    pass


def feature_selection(X, y, mode='univariate', model='regression'):
    """
    X: 2-D array of data
    y: 1-D array of labels
    mode: univariate | rfe | model_based | ranking
    """
    if mode == "univariate":
        # univariate feature selection
        # chi2, f_classif, mutual_info_classif
        from sklearn.feature_selection import chi2, f_classif
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.feature_selection import SelectFromModel
        # chi2_test, chi2_pval = chi2(X, y)
        # print("chi2 test {}".format(scores.shape))
        f_test, _ = f_classif(X, y)
        f_test = np.nan_to_num(f_test)
        f_test /= np.max(f_test)

        mi = mutual_info_classif(X, y)
        # mi = np.where(mi != np.nan)
        mi /= np.max(mi)

        plt.figure(figsize=(15, 5))
        for i in xrange(3):
            plt.subplot(1, 3, i + 1)
            plt.scatter(X[:, i], y)
            plt.xlabel("$x_{}$".format(i + 1), fontsize=14)
            if i == 0:
                plt.ylabel("$y$", fontsize=14)
                plt.title("F-test={:.2f}, MI={:.2f}"
                          .format(f_test[i], mi[i]),
                          fontsize=16)
                plt.show()

    if mode == "rfe":
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression, LassoCV
        # create the model and select 3 attributes
        if model == 'regression':
            clf = LogisticRegression()
        else:
            clf = LassoCV()
        rfe = RFE(clf, 10)
        rfe = rfe.fit(X, y)
        # summarize selection of attributes
        print("Support {}".format(rfe.support_.shape))
        print("Ranking {}".format(rfe.ranking_))
        return rfe

    if mode == "ranking":
        # feature ranking
        from sklearn.feature_selection import RFECV
        from sklearn.svm import SVR
        clf = SVR(kernel="linear")
        selector = RFECV(clf, step=1, cv=5)
        selector = selector.fit(X, y)
        print("features support {}".format(selector.support_))
        print("features ranking {}".format(selector.ranking_))

    if mode == "model_based":
        # Tree-based feature selection
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.feature_selection import SelectFromModel
        # fit an Extra Trees model to the data
        clf = ExtraTreesClassifier()
        clf.fit(X, y)
        # display the relative importance of each attribute
        # print(clf.feature_importances_)
        model = SelectFromModel(clf, prefit=True)
        X_new = model.transform(X)
        print(X_new.shape)
        # clf = Pipeline([
        #     ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
        #     ('classification', RandomForestClassifier())
        #    ])
        # clf.fit(X, y)

        # Mean decrease impurity
        from sklearn.ensemble import RandomForestClassifier
        # Load boston housing dataset as an example
        rf = RandomForestClassifier()
        rf.fit(X, y)
        print "Features sorted by their score:"
        features1 = sorted(map(lambda x: round(x, 4),
                               rf.feature_importances_),
                           reverse=True)

        # Mean decrease accuracy
        from sklearn.model_selection import ShuffleSplit
        from sklearn.metrics import r2_score
        from collections import defaultdict

        rf = RandomForestClassifier()
        scores = defaultdict(list)

        # crossvalidate the scores on a number of different random splits of
        # the data
        for tr_idx, te_idx in ShuffleSplit(n_splits=100, test_size=0.3,
                                           random_state=2017).split(X):
            trX, teX = X[tr_idx], X[te_idx]
            trY, teY = y[tr_idx], y[te_idx]
            rf.fit(trX, trY)
            acc = r2_score(teY, rf.predict(teX))
            for i in xrange(X.shape[1]):
                X_t = teX.copy()
                np.random.shuffle(X_t[:, i])
                shuff_acc = r2_score(teY, rf.predict(X_t))
                scores[i].append((acc - shuff_acc)/acc)
        print "Features sorted by their score:"
        features2 = sorted([(round(np.mean(score), 4), feat) for
                            feat, score in scores.items()], reverse=True)

        return features1, features2


def vis_cam(model, img, layer_name=None, penultimate_layer_idx=None):
    """
    params: img 4D tensor
    layer_name: layer whose feature map you want to visualize
    penultimate_layer_idx: previous layer index of layer_name whose grads
    are computed wrt its feature map
    """
    from vis.visualization import visualize_cam
    # from keras.models import load_model
    # model = load_model('./models/' + model_name)
    if layer_name is None:
        raise Warning("You need to provide a layer name indicating the layer"
                      " index of the cam you want to compute.")
        return -1
    layer_idx = [idx for idx, layer in enumerate(model.layers)
                 if layer.name == layer_name][0]
    pred_class = np.argmax(model.predict(np.expand_dims(img * 255, axis=0)))
    print("image shape {}, predicted_class = {}".format(img.shape,
                                                        pred_class))
    heatmap = visualize_cam(model, layer_idx, [pred_class], img,
                            penultimate_layer_idx)
    plt.imshow(heatmap)
    plt.show()
    plt.imsave(heatmap)


def plot_img_diff(orig_img, distorted, title):
    """ Helper function to display denoising """
    psnr, msqerr = mse(orig_img, distorted)
    sim = ssim(orig_img, distorted)
    plt.figure(figsize=(7, 4))
    plt.subplot(1, 3, 1)
    plt.title('Orignal Image')
    plt.imshow(orig_img, vmin=0, vmax=1, cmap=plt.cm.gray_r,
               interpolation='nearest')

    difference = distorted - orig_img
    plt.subplot(1, 3, 2)
    plt.title('Difference ($\ell_2$ norm: %.2f)'
              % np.sqrt(np.sum(difference ** 2)))
    plt.imshow(difference, vmin=-0.5, vmax=0.5, cmap=plt.cm.PuOr,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.subplot(1, 3, 3)
    plt.title('Distorted Image\nPSNR: {:.2f}\nMSE: {:.2f}\nSSIM: {:.2}'
              .format(psnr, msqerr, sim))
    plt.imshow(distorted, vmin=0, vmax=1, cmap='gray_r',
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.suptitle(title, size=16)
    plt.subplots_adjust(0.02, 0.02, 0.98, 0.79, 0.02, 0.2)


def denoising_dictionary_learning(img_orig, distorted, patch_size=(7, 7)):
    from sklearn.decomposition import MiniBatchDictionaryLearning
    from sklearn.feature_extraction.image import extract_patches_2d
    from sklearn.feature_extraction.image import reconstruct_from_patches_2d

    if img_orig.ndim == 3:
        h, w, c = img_orig.shape
    else:
        h, w = img_orig.shape

    print('Extracting reference patches... ')
    data = extract_patches_2d(img_orig, patch_size)
    data = data.reshape(data.shape[0], -1)
    intercept = np.mean(data, axis=0)
    data -= intercept
    data /= (np.std(data, axis=0) + 1e-5)
    # Learn the dictionary from reference patches
    print('Learning the dictionary...')
    dico = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=500)
    V = dico.fit(data).components_
    plt.figure(figsize=(5, 5))
    for i, comp in enumerate(V[:100]):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
        plt.suptitle('Dictionary learned from data patches\n' +
                     'Train on %d patches' % (len(data)),
                     fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    print('Extracting noisy patches... ')
    data = extract_patches_2d(distorted, patch_size)
    data = data.reshape(data.shape[0], -1)
    intercept = np.mean(data, axis=0)
    data -= intercept

    transform_algorithms = [
        ('OMP\n1 atom', 'omp',
         {'transform_n_nonzero_coefs': 1}),
        ('OMP\n38 atoms', 'omp',
         {'transform_n_nonzero_coefs': 38}),
        ('Least-angle regression\n15 atoms', 'lars',
         {'transform_n_nonzero_coefs': 15}),
        ('Thresholding\n alpha=0.1', 'threshold', {'transform_alpha': .1})
    ]
    print("Starting reconstruction of image from noisy patches...")
    reconstructions = {}
    for title, transform_algorithm, kwargs in transform_algorithms:
        print(title + '...')
        reconstructions[title] = img_orig.copy()
        dico.set_params(transform_algorithm=transform_algorithm, **kwargs)
        code = dico.transform(data)
        patches = np.dot(code, V)

        patches += intercept
        patches = patches.reshape(len(data), *patch_size)
        if transform_algorithm == 'threshold':
            patches -= patches.min()
            patches /= (patches.max() + 1e-5)
        reconstructions[title] = reconstruct_from_patches_2d(
            patches, (h, w))
        plot_img_diff(img_orig, reconstructions[title], title)
    plt.show()

    return reconstructions


def print_data_shapes(trX, trY, valX, valY, teX, teY):
    print("trX = {}, type = {}".format(trX.shape, trX.dtype))
    print("trY = {} type = {}".format(trY.shape, trY.dtype))
    print("valX = {}, type = {}".format(valX.shape, valX.dtype))
    print("valY = {}, type = {}".format(valY.shape, valY.dtype))
    print("teX = {}, type = {}".format(teX.shape, teX.dtype))
    print("teY = {}, type = {}".format(teY.shape, teY.dtype))


def wilson_score_interval(error, N):
    """
    Compute confidence intervals for classifier
    error +/- const * sqrt((error * (1 - error)) / n)
    param: const takes the following values
    1.64 (90%)
    1.96 (95%)
    2.33 (98%)
    2.58 (99%)
    param N: validation data set sample size
    Notice: this applies only to discrete-valued hypotheses. It assumes
    the sample S is drawn at random using teh same distribution from
    which future data will be drawn. Also it assumes data is independent
    of the hypothesis being tested.
    Provides only an approximate confidence interval. It's good when
    the sample contains at least 30 examples and error is not to close
    to 0 or 1.
    Notice that the confidence intervals on the classification error must
    be clipped to the values 0.0 and 1.0. It is impossible to have a
    negative error (e.g. less than 0.0) or an error more than 1.0.
    """
    err = error
    deviation = 1.96 * np.sqrt((error * (1 - error)) / N)
    strr = str(error) + '+/-' + str(deviation)
    print("error = {}".format(strr))
    print("There is a 95% likelihood that the confidence interval {}"
          " covers the true classification error of the model on unseen data."
          .format(np.clip(np.array([0.0, err + deviation]), 0, 1)))


def mcnerman_midp(b, c):
    """
    Compute McNemar's test using the "mid-p" variant suggested by:

    M.W. Fagerland, S. Lydersen, P. Laake. 2013. The McNemar test for
    binary matched-pairs data: Mid-p and asymptotic are better than exact
    conditional. BMC Medical Research Methodology 13: 91.

    `b` is the number of observations correctly labeled by the first---but
    not the second---system; `c` is the number of observations correctly
    labeled by the second---but not the first---system.
    The H0 hypothesis is that if midp value < 3.84 then the error rate
    for two classifiers if the same otherwise if midp value > 3.84 then
    error rate for the 2 classifiers is very different.
    """
    from scipy.stats import binom
    n = b + c
    x = min(b, c)
    dist = binom(n, .5)
    p = 2. * dist.cdf(x)
    midp = p - dist.pmf(x)
    if midp < 3.84:
        print("Error rate for classifiers is not significantly different")
    if midp > 3.84:
        print("Error rate for classifiers if significantly different")

    return midp
