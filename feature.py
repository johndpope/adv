# coding: utf-8


def robust_feature_selection(u, v):
    ###################################
    # 1. Univariate feature selection #
    ###################################
    # examines each feature individually to determine the strength of
    # the relationship of the feature with the response variable.
    # Particularly good for gaining a better understanding of data
    # (not necessarily for optimizing the feature set for better
    #  generalization).
    #######################
    # Pearson Correlation #
    #######################
    # simplest method for understanding a feature’s relation to the
    # response variable measures linear correlation between two variables
    # output range [-1, 1] : -1 => neg. corr., 1 => pos. corr., 0 => no corr.
    # If the relation is non-linear, Pearson correlation can be close to
    # zero even if there is a 1-1 correspondence between the two variables

    ##############################################################
    # Mutual Information & Maximal Information Coefficient (MIC) #
    ##############################################################
    # more robust option for correlation estimation which measures mutual
    # dependence between variables, typically in bits
    # I(X, Y) = \Sum(y in Y)\Sum(x in X)p(x, y)log(\frac{p(x, y)}{p(x)p(y)})
    # Can be inconvenient to use directly for feature ranking for 2 reasons.
    # 1st, it is not a metric and not normalized (i.e. doesn’t lie in a fixed
    #                                             range), so the MI values
    # can be incomparable between two datasets.
    # 2nd, it can be inconvenient to compute for continuous variables:
    # in general the variables need to be discretized by binning,
    # but the mutual information score can be quite sensitive to bin selection
    # (MIC) is a technique developed to address these shortcomings.
    # It searches for optimal binning and turns mutual information score
    # into a metric that lies in range [0;1]. There has been some critique
    # about MIC’s statistical power, i.e. the ability to reject the null
    # hypothesis when the null hypothesis is false. This may or may not
    # be a concern, depending on the particular dataset and its noisiness.
    from minepy import MINE
    m = MINE()
    x = np.random.uniform(-1, 1, 10000)
    m.compute_score(x, x**2)
    print m.mic()

    ########################
    # Distance Correlation #
    ########################
    # Another robust, competing method of correlation estimation is
    # distance correlation, designed explicitly to address the
    # shortcomings of Pearson correlation. While for Pearson correlation,
    # the correlation value 0 does not imply independence (e.g. x vs x**2),
    # distance correlation of 0 does imply that there is no dependence between
    # the two variables.
    # Two reasons why to prefer Pearson correlation over more sophisticated
    # methods such as MIC or distance correlation when the relationship
    # is close to linear. 1st, computing Pearson is much faster, may be
    # important in case of big datasets. 2nd, the range of the correlation
    # coefficient is [-1;1] (instead of [0;1] for MIC and distance correlation).
    # This can relay useful extra information on whether the relationship
    # is negative or positive, i.e. do higher feature values imply higher
    # values of the response variables or vice versa.
    # The question of negative versus positive correlation
    # is only well-posed if the relationship between the two variables
    # is monotonic.
    scipy.spatial.distance.correlation(u, v)  # u, v are 1-D vectors

    #######################
    # Model based ranking #
    #######################
    # One can use an arbitrary machine learning method to build a
    # predictive model for the response variable using each individual
    # feature, and measure the performance of each model. In fact, this
    # is already put to use with Pearson’s correlation coefficient, since
    # it is equivalent to standardized regression coefficient that is
    # used for prediction in linear regression. If the relationship between
    # a feature and the response variable is non-linear, there are a number
    # of alternatives, for example tree based methods (decision trees,
    #                                                  random forest),
    # linear model with basis expansion etc. Tree based methods are
    # probably among the easiest to apply, since they can model non-linear
    # relations well and don’t require much tuning. The main thing to avoid
    # is overfitting, so the depth of tree(s) should be relatively small,
    # and cross-validation should be applied.
    # Univariate feature selection is in general best to get a better
    # understanding of the data, its structure and characteristics.
    # It can work for selecting top features for model improvement in
    # some settings, but since it is unable to remove redundancy
    # (for example selecting only the best feature among a subset of
    #  strongly correlated features), this task is better left for other
    # methods.
    from sklearn.cross_validation import cross_val_score, ShuffleSplit
    from sklearn.datasets import load_boston
    from sklearn.ensemble import RandomForestRegressor

    # Load boston housing dataset as an example
    boston = load_boston()
    X = boston["data"]
    Y = boston["target"]
    names = boston["feature_names"]

    rf = RandomForestRegressor(n_estimators=20, max_depth=4)
    scores = []
    for i in range(X.shape[1]):
        score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
                                cv=ShuffleSplit(len(X), 3, .3))
        scores.append((round(np.mean(score), 3), names[i]))
    print sorted(scores, reverse=True)

    ################################################################
    # 2. Selecting good features. Linear models and reguralization #
    ################################################################
    # Many machine learning models have either some inherent internal
    # ranking of features or it is easy to generate the ranking from
    # the structure of the model. Using coefficients of regression models
    # for selecting and interpreting features. When all features are on
    # the same scale, the most important features should have the highest
    # coefficients in the model, while features uncorrelated with the
    # output variables should have coefficient values close to zero.
    # When there are multiple (linearly) correlated features
    # (as is the case with many real life datasets), the model becomes
    # unstable, meaning that small changes in the data can cause large
    # changes in the model (i.e. coefficient values), making model
    # interpretation very difficult (so called multicollinearity problem).


    from sklearn.linear_model import LinearRegression
    import numpy as np

    np.random.seed(0)
    size = 5000

    # A dataset with 3 features
    X = np.random.normal(0, 1, (size, 3))
    # Y = X0 + 2 * X1 + noise
    Y = X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 2, size)
    lr = LinearRegression()
    lr.fit(X, Y)

    # A helper method for pretty-printing linear models
    def pretty_print_linear(coefs, names=None, sort=False):
        if names is None:
            names = ["X%s" % x for x in range(len(coefs))]
            lst = zip(coefs, names)
            if sort:
                lst = sorted(lst, key=lambda x: -np.abs(x[0]))
                return " + ".join("%s * %s" % (round(coef, 3), name)
                                  for coef, name in lst)
    print "Linear model:", pretty_print_linear(lr.coef_)

    ######################
    # Reguralized Models #
    ######################

    # L1/Lasso
    # a method for adding additional constraints or penalty to a model,
    # with the goal of preventing overfitting and improving generalization.
    # Instead of minimizing a loss function E(X,Y), the loss function to
    # minimize becomes E(X,Y)+α‖w‖, where w is the vector of model
    # coefficients, ‖⋅‖ is typically L1 or L2 norm and α is a tunable
    # free parameter, specifying the amount of regularization. It forces
    # weak features to have zero as coefficients. Thus L1 regularization
    # produces sparse solutions, inherently performing feature selection
    # Note: L1 regularized regression is unstable in a similar way as
    # unregularized linear models are, the coefficients
    # (and thus feature ranks) can vary significantly even on small data
    # changes when there are correlated features.

    # Which brings us to L2 regularization.
    # adds the L2 norm penalty (α∑ni=1w2i) to the loss function.
    # Since the coefficients are squared in the penalty expression, it
    # has a different effect from L1-norm, namely it forces the
    # coefficient values to be spread out more equal.
    # The coefficients can vary widely for linear regression, depending
    # on the generated data. For L2 regularized model however, the
    # coefficients are quite stable and closely reflect how the data was
    # generated (all coefficients close to 1).
    # Lasso produces sparse solutions and as such is very useful selecting
    # a strong subset of features for improving model performance.
    # Ridge regression on the other hand can be used for data interpretation
    # due to its stability and the fact that useful features tend to have
    # non-zero coefficients. Since the relationship between the response
    # variable and features in often non-linear, basis expansion can be
    # used to convert features into a more suitable space, while keeping
    # the simple linear models fully applicable.

    ##################
    # Random forests #
    ##################
    # another popular approach for feature ranking
    # They provide two straightforward methods for feature selection:
    # mean decrease impurity and mean decrease accuracy.

    # Mean decrease impurity

    # Random forest consists of a number of decision trees. Every node
    # in the decision trees is a condition on a single feature, designed
    # to split the dataset into two so that similar response values end
    # up in the same set. The measure based on which the (locally)
    # optimal condition is chosen is called impurity. For classification,
    # it is typically either Gini impurity or information gain/entropy
    # and for regression trees it is variance. Thus when training a tree,
    # it can be computed how much each feature decreases the weighted
    # impurity in a tree. For a forest, the impurity decrease from each
    # feature can be averaged and the features are ranked according to
    # this measure.

    from sklearn.datasets import load_boston
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    # Load boston housing dataset as an example
    boston = load_boston()
    X = boston["data"]
    Y = boston["target"]
    names = boston["feature_names"]
    rf = RandomForestRegressor()
    rf.fit(X, Y)
    print "Features sorted by their score:"
    print sorted(zip(map(lambda x: round(x, 4),
                         rf.feature_importances_), names),
                 reverse=True)

    # Firstly, feature selection based on impurity reduction is biased
    # towards preferring variables with more categories
    # (see Bias in random forest variable importance measures).
    # Secondly, when the dataset has two (or more) correlated features,
    # then from the point of view of the model, any of these correlated
    # features can be used as the predictor, with no concrete preference
    # of one over the others. But once one of them is used, the importance
    # of others is significantly reduced since effectively the impurity
    # they can remove is already removed by the first feature. As a
    # consequence, they will have a lower reported importance. This is
    # not an issue when we want to use feature selection to reduce
    # overfitting, since it makes sense to remove features that are
    # mostly duplicated by other features. But when interpreting the
    # data, it can lead to the incorrect conclusion that one of the
    # variables is a strong predictor while the others in the same group
    # are unimportant, while actually they are very close in terms of
    # their relationship with the response variable

    # the difficulty of interpreting the importance/ranking of correlated
    # variables is not random forest specific, but applies to most model
    # based feature selection methods

    # Mean decrease accuracy

    # Directly measure the impact of each feature on accuracy of the model.
    # The general idea is to permute the values of each feature and measure
    # how much the permutation decreases the accuracy of the model.
    # Clearly, for unimportant variables, the permutation should have
    # little to no effect on model accuracy, while permuting important
    # variables should significantly decrease it.
    from sklearn.cross_validation import ShuffleSplit
    from sklearn.metrics import r2_score
    from collections import defaultdict

    X = boston["data"]
    Y = boston["target"]

    rf = RandomForestRegressor()
    scores = defaultdict(list)

    # crossvalidate the scores on a number of different random splits of
    # the data
    for train_idx, test_idx in ShuffleSplit(len(X), 100, .3):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        r = rf.fit(X_train, Y_train)
        acc = r2_score(Y_test, rf.predict(X_test))
        for i in range(X.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = r2_score(Y_test, rf.predict(X_t))
            scores[names[i]].append((acc-shuff_acc)/acc)
    print "Features sorted by their score:"
    print sorted([(round(np.mean(score), 4), feat) for
                  feat, score in scores.items()], reverse=True)

    # Keep in mind though that these measurements are made only after the
    # model has been trained (and is depending) on all of these features.
    # This doesn’t mean that if we train the model without one these feature,
    # the model performance will drop by that amount, since other,
    # correlated features can be used instead

    # Random forests are a popular method for feature ranking, since
    # they are so easy to apply: in general they require very little
    # feature engineering and parameter tuning and mean decrease impurity
    # is exposed in most random forest libraries. But they come with their
    # own gotchas, especially when data interpretation is concerned.
    # With correlated features, strong features can end up with low scores
    # and the method can be biased towards variables with many categories.
