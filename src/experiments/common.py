from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from data.domain import pack_domains
from time import time

def DCIclassify(source, target, s_pivots, t_pivots, dci, optimize=True):
    """
    This is a common routine for all experiments involving DCI. This function takes the source and target domain
    and packs them toghether in the dictionary-based format of DCI, learns the projection, trains the classifier
    (eventually optimizing hyperparameters), and evaluates the results. It also keeps track of some timings.
    :param source: the source domain (training + unlabeled); is an instance of Domain
    :param target: the source domain (testing  + unlabeled); is an instance of Domain
    :param s_pivots: a np.ndarray with the indexes of the source pivots
    :param t_pivots: a np.ndarray with the indexes of the target pivots
    :param dci: an instantiation of the DCI
    :param optimize: a boolean indicating whether to optimize the parameter C of the LinearSVC or not
    :return: (acc, dci_time, svm_time, test_time) where acc is the accuracy of classification measured in the test set
        dci_time is the time that the dci projection took, svm_time is the time to train (and optimize if requested)
        the classifier, and test_time is the time that the evaluation took
    """

    dX, dU, dP, dV = pack_domains(source, target, s_pivots, t_pivots)

    tinit = time()
    dLatent = dci.fit_transform(dU, dP, dX)
    dci_time = time() - tinit
    print('dci took {:.3f} seconds'.format(dci_time))

    Xs = dLatent[source.name()]
    Xt = dLatent[target.name()]

    svm = LinearSVC()
    if optimize:
        parameters = {'C': [10 ** i for i in range(-5, 5)]}
        svm = GridSearchCV(svm, parameters, n_jobs=-1, verbose=1, cv=5)

    tinit = time()
    svm.fit(Xs, source.y)
    svm_time = time() - tinit
    print('classification took {:.3f} seconds'.format(svm_time))
    if isinstance(svm, GridSearchCV):
        print('best_params {}'.format(svm.best_params_))
        svm = svm.best_estimator_

    # evaluation
    tinit = time()
    tyte_ = svm.predict(Xt)
    acc = (target.y == tyte_).mean()
    test_time = time() - tinit
    print('evaluation took {:.3f} seconds'.format(test_time))


    return acc, dci_time, svm_time, test_time