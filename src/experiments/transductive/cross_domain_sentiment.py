from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from data.tasks import MDS_task_generator
from data.domain import *
from os.path import abspath
from time import time
from classification.svmlight import SVMlight
from model.dci import DCI
from model.pivotselection import pivot_selection
import sys
import pandas as pd
import numpy as np
from util.results import Result


optimize = True
transductive = False
SVM = 'LinearSVC'#'SVMlight'
npivots = 1000
dcf= 'cosine'

dataset_home='../datasets/MDS'
result_file = '../results/cross-domain_normal.csv'
svmlight_home='/home/moreo/svm_light'
# results = Result.load(result_file)

assert transductive is False or SVM=='SVMlight', 'transduction is only available in SVMlight package'
assert SVM in ['SVMlight', 'LinearSVC'], 'unknown SVM, valid ones are "SVMlight" and "LinearSVC"'


nfolds=5
results = []
cv_accs = []
cv_taccs = []
for source, target, fold in MDS_task_generator(abspath(dataset_home), nfolds=nfolds):

    print('Pivot selection')
    tinit = time()
    s_pivots, t_pivots = pivot_selection(npivots, source.X, source.y, source.U, target.U,
                                         source.V, target.V,
                                         phi=1, show=min(20, npivots), cross_consistency=True)
    tend = time()-tinit
    print('pivot selection took {} seconds'.format(tend))

    dX, dU, dP, dV = pack_domains(source, target, s_pivots, t_pivots)

    print('DCI fit transform')
    dci = DCI(dcf=dcf, unify=False, post='normal')
    tinit = time()
    dLatent = dci.fit_transform(dU, dP, dX)
    tend = time()-tinit
    print('dci took {} seconds'.format(tend))

    print('Classification and test')
    Xs = dLatent[source.name()]
    Xt = dLatent[target.name()]
    T = Xt if transductive else None

    if SVM == 'SVMlight':
        svm = SVMlight(svmlightbase=svmlight_home, verbose=3, transduction=T)
    else:
        svm = LinearSVC()

    if optimize:
        parameters = {'C': [10 ** i for i in range(-5, 5)]}
        svm = GridSearchCV(svm, parameters, n_jobs=-1, verbose=1, cv=5)

    svm.fit(Xs, source.y)
    if isinstance(svm, GridSearchCV):
        print('best_params {}'.format(svm.best_params_))
        svm=svm.best_estimator_

    # evaluation
    print('Evaluation')

    if SVM=='SVMlight' and svm.is_transductive:
        tyte_ = svm.transduced_labels
        tacc = (target.y == tyte_).mean()
        cv_taccs.append(tacc)

    yte_ = svm.predict(Xt)
    acc = (target.y == yte_).mean()
    cv_accs.append(acc)
    print('fold-{}: {} {} : {:.3f}'.format(fold, source.name(), target.name(), acc))

    if fold == nfolds-1:
        acc_ave = np.mean(cv_accs)
        if len(cv_taccs)>0:
            acc_tave = np.mean(cv_taccs)
            results.append((source.name(), target.name(), acc_ave, acc_tave))
            print('Average {} {} : {:.3f} {:.3f}'.format(source.name(), target.name(), acc_ave, acc_tave))
        else:
            results.append((source.name(), target.name(), acc_ave))
            print('Average {} {} : {:.3f}'.format(source.name(), target.name(), acc_ave))
        cv_accs = []
        cv_taccs = []


print('Transductive {}, optimization {}, pivots={}, dcf={}'.format(transductive, optimize, npivots, dcf))
result_file = '{}_{}_{}_{}.txt'.format('MDS',transductive, optimize, npivots)
with open(result_file, 'wt') as fo:
    if SVM=='SVMlight' and svm.is_transductive:
        for s,t,acc,tacc in results:
            print('{}\t{}\t{:.3f}\t{:.3f}'.format(s,t,acc,tacc))
            fo.write('{}\t{}\t{:.3f}\t{:.3f}\n'.format(s, t, acc,tacc))

    else:
        for s,t,acc in results:
            print('{}\t{}\t{:.3f}'.format(s,t,acc))
            fo.write('{}\t{}\t{:.3f}\n'.format(s, t, acc))
