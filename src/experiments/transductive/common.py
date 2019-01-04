from sklearn.model_selection import GridSearchCV
from classification.svmlight import SVMlight
from data.domain import pack_domains
from time import time

def DCItransduction(source, target, s_pivots, t_pivots, dci, svmlight_home, optimize=False, transductive=True):

    dX, dU, dP, dV = pack_domains(source, target, s_pivots, t_pivots)

    print('DCI fit transform')
    tinit = time()
    dLatent = dci.fit_transform(dU, dP, dX)
    dci_time = time() - tinit
    print('dci took {} seconds'.format(dci_time))

    print('Classification and test')
    Xs = dLatent[source.name()]
    Xt = dLatent[target.name()]

    T = Xt if transductive else None
    svm = SVMlight(svmlightbase=svmlight_home, verbose=0, transduction=T)

    if optimize:
        parameters = {'C': [10 ** i for i in range(-5, 5)]}
        svm = GridSearchCV(svm, parameters, n_jobs=-1, verbose=1, cv=5)

    tinit = time()
    svm.fit(Xs, source.y)
    svm_time = time() - tinit
    if isinstance(svm, GridSearchCV):
        print('best_params {}'.format(svm.best_params_))
        svm = svm.best_estimator_

    # evaluation
    print('Evaluation')
    tinit = time()
    if svm.is_transductive:
        tyte_ = svm.transduced_labels
    else:
        tyte_ = svm.predict(Xt)
    acc = (target.y == tyte_).mean()
    test_time = time() - tinit

    return acc, dci_time, svm_time, test_time