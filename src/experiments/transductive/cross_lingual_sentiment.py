from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from data.tasks import WebisCLS10_task_generator
from time import time
from data.domain import pack_domains
from classification.svmlight import SVMlight
from model.dci import DCI
from model.pivotselection import pivot_selection
import os

optimize = True
transductive = False
#SVM = 'SVMlight'
SVM = 'LinearSVC'
dcf='linear'
npivots = 450
dataset_home='../datasets/Webis-CLS-10'
svmlight_home='/home/moreo/svm_light'

assert transductive is False or SVM=='SVMlight', 'transduction is only available in SVMlight package'
assert SVM in ['SVMlight', 'LinearSVC'], 'unknown SVM, valid ones are "SVMlight" and "LinearSVC"'

results = []

for source, target, oracle in WebisCLS10_task_generator(os.path.abspath(dataset_home)):

    # pivot selection
    s_pivots, t_pivots = pivot_selection(npivots, source.X, source.y, source.U, target.U,
                                         source.V, target.V,
                                         oracle=oracle, phi=30, show=min(20, npivots), cross_consistency=False)

    # document representation
    dX, dU, dP, dV = pack_domains(source, target, s_pivots, t_pivots)

    dci = DCI(dcf=dcf, unify=True, post='normal')
    tinit = time()
    dLatent = dci.fit_transform(dU, dP, dX, dV)
    tend = time() - tinit
    print('dci took {} seconds'.format(tend))

    # classification
    print('Training the classifier')

    Xs = dLatent[source.name()]
    Xt = dLatent[target.name()]
    T  = Xt if transductive else None

    if SVM == 'SVMlight':
        svm = SVMlight(svmlightbase=svmlight_home, verbose=0, transduction=T)
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

    if SVM == 'SVMlight' and svm.is_transductive:
        tyte_ = svm.transduced_labels
        tacc = (target.y == tyte_).mean()

    yte_ = svm.predict(Xt)
    acc = (target.y==yte_).mean()

    if SVM == 'SVMlight' and svm.is_transductive:
        print('{} pivots: {} {} : {:.3f} {:.3f}'.format(npivots, source.name(), target.name(), acc, tacc))
    else:
        print('{} pivots: {} {} : {:.3f}'.format(npivots, source.name(), target.name(), acc))

    results.append((source.name()+'_'+target.name(), acc))

print('Transductive {}, optimization {}, pivots={}, dcf={}'.format(transductive, optimize, npivots, dcf))
for d,r in results:
    print('{}\t{:.3f}'.format(d,r))



