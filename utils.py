import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt


def rank_classifiers(models, X, Y, nb_epochs=2, batch_size=128):
    """
    models: list of tuples [('model name', model object), ...]
    X: training data
    Y: labels, should be one-hot and not clipped
    """
    cv_results = []
    results = []
    names = []
    counter = 1
    skf = StratifiedKFold(n_splits=10, random_state=2017, shuffle=True)
    for name, model in models:
        print("\n\nTesting model {}\n".format(name))
        model.summary()
        for tr_id, te_id in skf.split(X, np.argmax(Y, axis=1)):
            print("Fold {}".format(counter))
            trX, teX = X[tr_id], X[te_id]
            trY, teY = Y[tr_id], Y[te_id]
            if name == "mlp":
                trX = trX.reshape(-1, 784)
                teX = teX.reshape(-1, 784)
            elif name == "irnn":
                trX = trX.reshape(-1, 784, 1)
                teX = teX.reshape(-1, 784, 1)
            model.fit(trX, trY, nb_epoch=nb_epochs, batch_size=batch_size,
                      validation_split=0.2, verbose=1)
            teY_pred = model.predict(teX)
            scores = model.evaluate(teX, teY, verbose=0)
            report = classification_report(np.argmax(teY, axis=1),
                                           np.argmax(teY_pred, axis=1))
            print(report)
            cv_results.append(scores[1] * 100)
            counter += 1
        results.append([np.mean(cv_results), np.std(cv_results)])
        names.append(name)
        print("\nmodel = {}, mean = {}, std = {}"
              .format(name, np.mean(results), np.std(results)))
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    plt.ylabel('Accuracy')
    ax.set_xticklabels(names)
    plt.show()


def plot_mnist(X, y, X_embedded, name, min_dist=10.0):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(frameon=False)
    plt.title("\\textbf{MNIST dataset} -- Two-dimensional "
              "embedding of 60,000 handwritten digits with %s" % name)
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
