import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from matplotlib import pyplot as plt


def rank_classifiers(models, X, Y):
    results = []
    names = []
    for name, model in models:
        kf = KFold(n_splits=10, random_state=2017)
        cv_results = cross_val_score(model, X, Y, cv=kf, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print("model = {}, mean = {}, std = {}".format(name,
                                                       cv_results.mean(),
                                                       cv_results.std()))
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
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
