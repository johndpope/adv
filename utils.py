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
# from cleverhans.utils import pair_visual
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


def rank_classifiers(models, X, Y, X_test, Y_test, X_test_adv,
                     save_model=False, epochs=2, batch_size=128,
                     pretrained=False):
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
    img_row, img_col, img_chn = X.shape[1:]
    X = X.reshape(-1, np.prod(X.shape[1:]))
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
            if name.lower() == "mlp":
                trX = trX.reshape(-1, img_row * img_col * img_chn)
                teX = teX.reshape(-1, img_row * img_col * img_chn)
                X_test = X_test.reshape(-1, img_row * img_col * img_chn)
                X_test_adv = X_test_adv.reshape(-1,
                                                img_row * img_col * img_chn)
            elif name.lower() == "irnn":
                trX = trX.reshape(-1, img_row * img_col, img_chn)
                teX = teX.reshape(-1, img_row * img_col, img_chn)
                X_test = X_test.reshape(-1, img_row * img_col, img_chn)
                X_test_adv = X_test_adv.reshape(-1, img_row * img_col, img_chn)
            else:
                trX = trX.reshape(-1, img_row, img_col, img_chn)
                teX = teX.reshape(-1, img_row, img_col, img_chn)
                X_test = X_test.reshape(-1, img_row, img_col, img_chn)
                X_test_adv = X_test_adv.reshape(-1, img_row, img_col, img_chn)
            if not pretrained:
                model.fit(trX, trY, nb_epoch=epochs, batch_size=batch_size,
                          validation_split=0.2, verbose=1)
            # print("Dataset:\ntrX: {}\nteX: {}\nX_test: {}\ntrY: {}\n teY: {}"
            #       .format(trX.shape, teX.shape, X_test.shape, trY.shape,
            #               teY.shape))
            scores = model.evaluate(teX, teY, verbose=0)
            teY_pred = model.predict(teX)
            cv_results.append(scores[1])
            mean_tpr, mean_fpr = roc_auc(teY, teY_pred, counter,
                                         color, mean_tpr, mean_fpr)
            counter += 1
        counter = 0
        legit_scores = model.evaluate(X_test, Y_test)
        # adv_scores = model.evaluate(X_test_adv, Y_test)
        print("\nmodel = {}, mean = {}, std = {}, legit test acc. = {}, "
              "adv. test acc. = "
              .format(name, np.mean(cv_results), np.std(cv_results),
                      legit_scores[1]))
        # results.append([np.mean(cv_results), np.std(cv_results)])
        results.append(cv_results)
        cv_results = []
        names.append(name)
        teY_pred = model.predict(teX)
        report = classification_report(np.argmax(teY, axis=1),
                                       np.argmax(teY_pred, axis=1))
        print(report)
        plot_roc_auc(X, Y, skf, mean_tpr, mean_fpr)
        if save_model:
            model.save("models/{}.hdf5".format(name))

    boxplot(results, names)


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Classifiers')


def boxplot(results, names):
    # boxplot algorithm comparison
    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4),
    # sharey=True)
    results *= 100
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax1 = fig.add_subplot(121)
    ax1.set_ylabel('Accuracy %')
    # add patch_artist=True option to ax.boxplot() to get fill color
    bp = ax1.boxplot(results, patch_artist=True)

    # change outline color, fill color and linewidth of the boxes
    colors = ['#808000', '#0000FF', '#008080', '#FF00FF',
              '#008000', '#000080', '#800080', '#00FF00', '#00FFFF']

    for idx, box in enumerate(bp['boxes']):
        # change outline color
        box.set(color='#000000', linewidth=2)
        # change fill color
        # if len(names) > len(colors):
        #     box.set(facecolor=colors[0])
        if idx <= len(names) - 1:
            box.set(facecolor=colors[idx])
        else:
            box.set(facecolor=colors[0])

    # change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)

    # change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)

    # change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)

        # change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)

    ax2 = fig.add_subplot(122)
    parts = ax2.violinplot(results, showmeans=False, showextrema=False,
                           showmedians=False)
    # ax1.set_xticklabels([])
    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    quartile1, medians, quartile3 = np.percentile(results, [25, 50, 75],
                                                  axis=1)
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(results, quartile1, quartile3)])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax2.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)
    ax2.set_yticklabels([])

    for ax in [ax1, ax2]:
        set_axis_style(ax, names)

    # plt.subplots_adjust(left=0.0, bottom=0.0, right=1.5, top=0.9,
    #                     wspace=0.0, hspace=0.0)
    plt.show()


def plot_2d_embedding(X, y, X_embedded, name, img_row=28, img_col=28,
                      img_chn=1, min_dist=10.0):
    fig = plt.figure(figsize=(3, 3))
    ax = plt.axes(frameon=False)
    plt.title("2D embedding of %s" % name)
    plt.setp(ax, xticks=(), yticks=())
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                        wspace=0.0, hspace=0.0)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, marker="x")
    plt.colorbar()

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
            if img_chn == 1:
                datum = X[i].reshape(img_row, img_col)
            if img_chn == 3:
                datum = X[i].reshape(img_row, img_col, img_chn)
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(datum,
                                      cmap=plt.cm.gray_r),
                X_embedded[i])
            ax.add_artist(imagebox)
    plt.show()


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


def visualize_cmap(model, img, layer_name="conv5_3", target_class=1,
                   output_path='./gradcam/gradimg.png'):
    width, height = img.shape[:2]

    # Get the weights and biases of the softmax.
    class_weights = model.layers[-1].get_weights()[0]
    class_bias = model.layers[-1].get_weights()[1]
    print("class weights shape: {}".format(class_weights.shape))
    print("class bias shape: {}".format(class_bias.shape))
    # weights_shape = class_weights.shape
    # class_weights = class_weights.reshape(class_weights.shape[0],
    #                                       np.prod(class_weights.shape[1:]))
    # class_weights = normalize(class_weights)
    # class_weights = class_weights.reshape(weights_shape)
    final_conv_layer = get_output_layer(model, layer_name)
    get_output = K.function([model.layers[0].input, K.learning_phase()],
                            [final_conv_layer.output,
                             model.layers[-1].output])
    [conv_outputs, predictions] = get_output([np.expand_dims(img, axis=0), 0])
    print("predictions from K.function: {}".format(np.argmax(predictions,
                                                             axis=1)))
    preds = model.predict(np.expand_dims(img, axis=0))
    predicted_class = np.argmax(preds)
    prob_predicted_class = np.max(preds, axis=1)
    print("True label {}, predicted label {}, with probability {}"
          .format(target_class,
                  predicted_class,
                  prob_predicted_class)
          )
    conv_outputs = conv_outputs[0, :, :, :]

    # Create the class activation map.
    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[:2])
    counter = 0
    for w in class_weights[:, target_class]:
        if counter > conv_outputs.shape[2] - 1:
            counter = 0
        cam += w * conv_outputs[:, :, counter]
        counter += 1
    cam /= (np.max(cam) + 1e-5)
    cam = cv2.resize(cam, (height, width))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    img = heatmap * 0.5 + img
    # img = np.squeeze(img)
    cv2.imwrite(output_path, img)
    # plt.imshow(img)
    # plt.show()
    print("image shape: {}".format(img.shape))
    return img


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


def find_top_predictions(model, teX, teY, teX_adv, count, img_row=28,
                         img_col=28, img_chn=1, random=False):
    if random:
        rand_ind = np.unique(np.random.randint(0, len(teX), count))
        preds = model.predict(teX[rand_ind])
        orig_targets = np.argmax(teY[rand_ind], axis=1)
    else:
        orig_labels = np.argmax(teY, axis=1)
        preds = model.predict(teX)
        adv_preds = model.predict(teX_adv)

    predicted_labels = np.argmax(preds, axis=1)
    adv_predicted_labels = np.argmax(adv_preds, axis=1)
    accuracies = np.max(preds, axis=1)
    adv_accuracies = np.max(adv_preds, axis=1)
    final_matrix = np.zeros((len(adv_preds), 6))
    final_matrix[:, 0] = np.arange(len(adv_preds))  # tracking indeces for images
    final_matrix[:, 1] = orig_labels
    final_matrix[:, 2] = predicted_labels
    final_matrix[:, 3] = adv_predicted_labels
    final_matrix[:, 4] = accuracies
    final_matrix[:, 5] = adv_accuracies
    # sort the array by row based on accuracies
    final_matrix.view('i8, i8, i8, i8, i8, i8').sort(order=['f5'], axis=0)
    # sort accuracies in descending order, keep only K-top
    # final_matrix = final_matrix[::-1][:count]
    final_matrix = final_matrix[::-1]
    sorted_adv_samples = np.zeros_like(final_matrix)
    for idx, entry in enumerate(final_matrix):
        if entry[1] == entry[2] and entry[4] > 0.8 and \
           entry[2] != entry[3] and entry[5] > 0.8:
            sorted_adv_samples[idx] = entry

    # remove all zeros rows
    sorted_adv_samples = sorted_adv_samples[
        ~(sorted_adv_samples == 0).all(axis=1)
    ]
    print("#{} top adv. samples found".format(sorted_adv_samples.shape[0]))
    if count <= sorted_adv_samples.shape[0]:
        # select unique predictd labels
        # _, ind = np.unique(final_matrix[:, 2], return_index=True)
        # select top k adv. samples
        final_matrix = sorted_adv_samples[:count]
        imgs = teX[np.int32(final_matrix[:, 0])]
        imgs_adv = teX_adv[np.int32(final_matrix[:, 0])]
        orig_targets = np.int32(final_matrix[:, 1])
        targets = np.int32(final_matrix[:, 2])
        adv_targets = np.int32(final_matrix[:, 3])
        accuracies = final_matrix[:, 4]
        adv_accuracies = final_matrix[:, 5]
        print("predicted top {} image labels: {}, true labels: {}\naccuracy: {}"
              .format(count, targets, orig_targets, accuracies))
        print("top {} adv. image accuracy: {}\nadv. labels: {}"
              .format(count, adv_accuracies, adv_targets))
        print("top predicted images:")
        if img_chn == 3:
            shape = img_row, img_col, img_chn
        if img_chn == 1:
            shape = img_row, img_col
        fig, axes = plt.subplots(2, len(imgs))
        fig.subplots_adjust(top=1, right=2)
        for im in xrange(len(imgs)):
            axes[0][im].imshow(imgs[im].reshape(shape))
            axes[0][im].set_title("Actual label: {}\nPredicted label: {}"
                                  "\nProb. {:.4}"
                                  .format(orig_targets[im],
                                          targets[im],
                                          accuracies[im] * 100))
            axes[0][im].axis('off')
            axes[1][im].imshow(imgs_adv[im].reshape(shape))
            axes[1][im].set_title("Predicted label: {}\nProb.: {:.4}"
                                  .format(adv_targets[im],
                                          adv_accuracies[im] * 100))
            axes[1][im].axis('off')
        plt.show()

        return np.float32(np.array(imgs)), final_matrix[:, 0]
    else:
        print("Reduce the number of adv. samples.")

    # for key, val in enumerate(imgs_adv):
    #     pair_visual(imgs[key].reshape(28, 28),
    #                 imgs_adv[key].reshape(28, 28))
    #     run_gradcam(model, model_name, imgs[key], orig_targets[key],
    #                 layer_name)


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

    # np.max(imgA) should be 255.
    psnr = 20.0 * np.log10(np.max(imgA) + 1e-6) - 10.0 * np.log10(err + 1e-6)

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


def vis_cam(model, img, layer_name=None, penultimate_layer_idx=None,
            mode='saliency', nb_out_imgs=5):
    """
    params: img 3D tensor
    layer_name: layer whose feature map you want to visualize
    penultimate_layer_idx: previous layer index of layer_name whose grads
    are computed wrt its feature map
    """
    if layer_name is None:
        raise Warning("You need to provide a layer name indicating the layer"
                      " index of the cam you want to compute.")
        return -1

    layer_idx = [idx for idx, layer in enumerate(model.layers)
                 if layer.name == layer_name][0]

    # Notice the difference of images for each separate method
    # Img: for cam has to be 4D
    #      for saliency has to be 3D
    #      for dense no img param required
    #      for conv no img param required
    pred_class = np.argmax(model.predict(np.expand_dims(img, axis=0)))
    print("image shape {}, predicted_class = {}".format(img.shape,
                                                        pred_class))

    if img is not None:
        if img.ndim == 4:
            _, w, h, c = img.shape
        elif img.ndim == 3:
            w, h, c = img.shape

        if np.max(img) != 255:
            img *= 255
            print("Image max pixel value: {}".format(np.max(img)))

    if mode == 'saliency':
        from vis.visualization import visualize_saliency
        heatmap = visualize_saliency(model, layer_idx,
                                     list(np.arange(nb_out_imgs)),
                                     img)
        if heatmap.shape[2] == 1:
            heatmap = heatmap.reshape(heatmap.shape[0], heatmap.shape[1])
        # elif heatmap.shape[3] == 3:
        #     heatmap = heatmap.reshape(heatmap.shape[1:])
        plt.imshow(heatmap)
        plt.show()
    if mode == 'cam':
        from vis.visualization import visualize_cam
        heatmap = visualize_cam(model, layer_idx, list(np.arange(nb_out_imgs)),
                                np.expand_dims(img, axis=0),
                                penultimate_layer_idx)
        if heatmap.shape[3] == 1:
            heatmap = heatmap.reshape(heatmap.shape[1], heatmap.shape[2])
        elif heatmap.shape[3] == 3:
            heatmap = heatmap.reshape(heatmap.shape[1:])
        plt.imshow(heatmap)
        plt.show()
    if mode == 'conv':
        from vis.utils import utils
        from vis.visualization import visualize_activation, get_num_filters
        # Visualize all filters in this layer.
        filters = np.arange(get_num_filters(model.layers[layer_idx]))
        # Generate input image for each filter. Here `text` field is
        # used to overlay `filter_value` on top of the image.
        vis_images = []
        for idx in filters:
            img = visualize_activation(model, layer_idx, filter_indices=idx)
            # img = utils.draw_text(img, str(idx))
            vis_images.append(img)

        # Generate stitched image palette with 8 cols.
        stitched = utils.stitch_images(vis_images, cols=8)
        if stitched.shape[2] == 1:
            stitched = stitched.reshape(-1, stitched.shape[1])
        plt.figure(figsize=(60, 30))
        plt.axis('off')
        plt.imshow(stitched)
        plt.title(layer_name)
        plt.show()
    if mode == 'dense':
        from vis.visualization import visualize_activation
        from vis.utils import utils
        # Generate three different images of the same output index.
        del img
        vis_images = []
        for idx in xrange(nb_out_imgs):
            img = visualize_activation(model, layer_idx,
                                       filter_indices=list(np.arange(nb_out_imgs)),
                                       max_iter=500)
            # img = utils.draw_text(img.reshape(28, 28), str(pred_class))
            vis_images.append(img)

        stitched = utils.stitch_images(vis_images)
        if stitched.shape[2] == 1:
            stitched = stitched.reshape(-1, stitched.shape[1])
        plt.figure(figsize=(60, 30))
        plt.axis('off')
        plt.imshow(stitched)
        plt.title(layer_name)
        plt.show()


def plot_img_diff(orig_img, distorted, title):
    """ Helper function to display denoising """
    psnr, msqerr = mse(orig_img, distorted)
    sim = ssim(orig_img, distorted)
    plt.figure(figsize=(7, 4))
    plt.subplot(1, 3, 1)
    plt.title('Orignal Image')
    plt.imshow(orig_img, vmin=0, vmax=1,
               interpolation='nearest')

    difference = distorted - orig_img
    plt.subplot(1, 3, 2)
    plt.title('Difference ($\ell_2$ norm: %.2f)'
              % np.sqrt(np.sum(difference ** 2)))
    plt.imshow(difference, vmin=-0.5, vmax=0.5, cmap=plt.cm.inferno,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.subplot(1, 3, 3)
    plt.title('Distorted Image\nPSNR: {:.2f}\nMSE: {:.2}\nSSIM: {:.2}'
              .format(psnr, msqerr, sim))
    plt.imshow(distorted, vmin=0, vmax=1,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.suptitle(title, size=16)
    plt.subplots_adjust(0.02, 0.02, 0.98, 0.79, 0.02, 0.2)


def denoising_dictionary_learning(img_orig, distorted, patch_size=(7, 7)):
    from sklearn.decomposition import MiniBatchDictionaryLearning
    from sklearn.feature_extraction.image import extract_patches_2d
    from sklearn.feature_extraction.image import reconstruct_from_patches_2d

    if img_orig.ndim == 3 and img_orig.shape[2] == 3:
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
    print("max: {}, min: {}".format(np.max(trX), np.min(trX)))


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


def tsne(X):
    from MulticoreTSNE import MulticoreTSNE as TSNE
    tsne = TSNE(n_jobs=-1)
    X_embedded = tsne.fit_transform(
        X.reshape(X.shape[0], -1).astype(np.float64)
    )

    return X_embedded


def extract_hypercolumns(model, layer_indexes, image):
    layers = [model.layers[layer].output for layer in layer_indexes]
    get_feature = K.function([model.layers[0].input, K.learning_phase()],
                             layers)
    feature_maps = get_feature([np.expand_dims(image, axis=0),  0])
    hypercolumns = []
    for convmap in feature_maps:
        fmaps = [np.float32(convmap[0, :, :, i]) for i in xrange(convmap.shape[-1])]
        layer = []
        for fmap in fmaps:
            fmap = np.abs(fmap)
            norm = np.max(np.max(fmap, axis=0), axis=0)
            if norm > 0:
                fmap = fmap / norm
                upscaled = scipy.misc.imresize(fmap, size=(image.shape[0],
                                                           image.shape[1]),
                                               mode='F',
                                               interp='bilinear')
                layer.append(upscaled)

        hypercolumns.append(np.mean(np.float32(layer), axis=0))

    return np.asarray(hypercolumns)


def visualize_hypercolumns(model, img, layers_extract=[2]):

    # img = np.float32(cv2.resize(original_img, (200, 66))) / 255.0
    original_img = img * 255.
    hc = extract_hypercolumns(model, layers_extract, img)
    avg = np.product(hc, axis=0)
    avg = np.abs(avg)
    avg = avg / np.max(np.max(avg))

    heatmap = cv2.applyColorMap(np.uint8(255 * avg), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / np.max(np.max(heatmap))
    heatmap = cv2.resize(heatmap, original_img.shape[0:2][::-1])

    both = 255 * heatmap * 0.7 + original_img
    both = both / np.max(both)

    return np.squeeze(both)


def visualize_occlussion_map(model, img):
    imgs, windows = [], []
    # img = cv2.resize(original_img, (200, 66))
    original_img = img * 255.
    base_angle = model.predict(np.expand_dims(img, axis=0))

    for x in xrange(0, img.shape[1], 2):
        for y in xrange(0, img.shape[0], 2):
            windows.append((x, y, 15, 15))
            windows.append((x, y, 50, 50))

    for window in windows:
        x, y, w, h = window
        masked = img * 1
        masked[y: y + h, x: x + w] = 0
        imgs.append(masked)

    angles = model.predict(np.array(imgs))
    result = np.zeros(shape=img.shape[:2], dtype=np.float32)
    import pdb; pdb.set_trace() ## DEBUG ##
    for i, window in enumerate(windows):
        diff = np.abs(angles[i] - base_angle)
        x, y, w, h = window
        result[y: y + h, x: x + w] += diff
    mask = np.abs(result)
    mask = mask / np.max(np.max(mask))
    # mask[np.where(mask < np.percentile(mask, 60))] = 0
    mask = cv2.resize(mask, original_img.shape[0:2][::-1])

    result = original_img
    result[np.where(mask == 0)] = 0

    return result


def l2(x):
    noisy = x + K.random.normal(shape=x.shape)
    difference = noisy - x
    error = K.sqrt(K.sum(difference ** 2))

    return error


def plot_kde(X):
    from scipy import stats
    if X.ndim > 1:
        flat = X.flatten()
    my_pdf = stats.gaussian_kde(flat)
    x = np.linspace(-5, 5, 100)
    # plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=0.9,
    #                     wspace=0.0, hspace=0.0)
    plt.subplot(121)
    residuals = stats.probplot(flat, plot=plt)
    # plt.text(1, -0.5, u'$Rˆ2$ = {:.2}'.format(residuals[1][2]), fontsize=12)
    plt.subplot(122)
    plt.plot(x, my_pdf(x), 'r')  # distribution function
    result = plt.hist(flat, normed=1, alpha=0.3)
    plt.show()


def qqplot():
    import pandas as pd
    data = pd.read_clipboard(header=None).values.flatten()
    data.sort()
    norm = np.random.normal(0, 2, len(data))
    norm.sort()
    plt.figure(figsize=(12, 8), facecolor='1.0')

    plt.plot(norm, data, "o")

    #generate a trend line as in http://widu.tumblr.com/post/43624347354/matplotlib-trendline
    z = np.polyfit(norm, data, 1)
    p = np.poly1d(z)
    plt.plot(norm, p(norm), "k--", linewidth=2)
    plt.title("Normal Q-Q plot", size=28)
    plt.xlabel("Theoretical quantiles", size=24)
    plt.ylabel("Expreimental quantiles", size=24)
    plt.tick_params(labelsize=16)
    plt.show()


def fit_normal():
    from scipy.stats import norm
    from numpy import linspace
    from pylab import plot, show, hist, title

    # picking 150 of from a normal distrubution
    # with mean 0 and standard deviation 1
    samp = norm.rvs(loc=0, scale=1, size=150)
    param = norm.fit(samp)  # distribution fitting

    # now, param[0] and param[1] are the mean and
    # the standard deviation of the fitted distribution
    x = linspace(-5, 5, 100)
    # fitted distribution
    pdf_fitted = norm.pdf(x, loc=param[0], scale=param[1])
    # original distribution
    pdf = norm.pdf(x)

    title('Normal distribution')
    plot(x, pdf_fitted, 'r-', x, pdf, 'b-')
    hist(samp, normed=1, alpha=0.3)
    show()


def plot_classifier_boundary(X, y, models, dataset_name='mnist'):
    import os
    from itertools import product
    X = tsne(X)
    print("X.shape: {}".format(X.shape))
    # create a mesh to plot in
    x_min, x_max = X[:100, 0].min() - 1, X[:100, 0].max() + 1
    y_min, y_max = X[:100, 1].min() - 1, X[:100, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                         np.arange(y_min, y_max, 0.2))
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    fig, ax = plt.subplots(2, int(len(models)/2 + 1))
    for idx, model in zip(product([0, 1], [0, 1, 2]), models):
        model[1].fit(X[:100], y[:100], shuffle=True,
                     validation_split=0.1, epochs=10,
                     batch_size=64, verbose=1)
        Z = np.argmax(model[1].predict(np.c_[xx.ravel(), yy.ravel()]), axis=1)

        # Put the result into a color plot
        # Z = Z.reshape(xx.shape)
        Z = Z.reshape(xx.shape)
        if int(len(models) / 2) + 1 <= 1:
            ax[idx[0]].contourf(xx, yy, Z, cmap=plt.cm.Paired)
            ax[idx[0]].axis('off')
            ax[idx[0]].scatter(X[:100, 0], X[:100, 1],
                               c=np.argmax(y, axis=1)[:100],
                               cmap=plt.cm.Paired)
            ax[idx[0]].set_title(model[0])
        else:
            ax[idx[0], idx[1]].contourf(xx, yy, Z, cmap=plt.cm.Paired)
            ax[idx[0], idx[1]].axis('off')

            # Plot also the training points
            ax[idx[0], idx[1]].scatter(X[:100, 0], X[:100, 1],
                                       c=np.argmax(y, axis=1)[:100],
                                       cmap=plt.cm.Paired)
            # also plot adversarial data
            # path = 'adv_data/' + str(model[0].lower()) + \
            #        dataset_name.lower() + '_adv.npy'
            # if os.path.exists(path):
            #     X_adv = np.load(path)
            #     ax[idx[0], idx[1]].scatter(X_adv[:, 0], X[:, 1],
            #                                c=np.argmax(y, axis=1),
            #                                cmap=plt.cm.Paired)

            ax[idx[0], idx[1]].set_title(model[0])

    plt.show()
