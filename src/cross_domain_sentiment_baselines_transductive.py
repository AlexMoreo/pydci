from sklearn.model_selection import cross_val_score
from data.tasks import MDS_task_generator, UpperMDS_task_generator
from data.domain import *
from os.path import abspath
from classification.svmlight import SVMlight
from util.results import Result


dataset_home='../datasets/MDS'
svmlight_home='../../svm_light'

nfolds=5

results = Result(['dataset', 'task', 'method', 'acc'])
for domain in UpperMDS_task_generator(abspath(dataset_home)):
    isvm = SVMlight(svmlightbase=svmlight_home, verbose=0, transduction=None)
    score = cross_val_score(isvm, domain.X, domain.y, cv=nfolds).mean()
    results.add(dataset='MDS', task=domain.domain, method='UPPER', acc=score)
results.pivot(grand_totals=True)

results = Result(['dataset', 'task', 'method', 'fold', 'acc'])
for source, target, fold, task in MDS_task_generator(abspath(dataset_home), nfolds=nfolds):
    source_name = source.domain
    target_name = target.domain
    source, target = unify_feat_space(source, target)

    isvm = SVMlight(svmlightbase=svmlight_home, verbose=0, transduction=None).fit(source.X, source.y)
    tsvm = SVMlight(svmlightbase=svmlight_home, verbose=0, transduction=target.X).fit(source.X, source.y)

    yte_ = isvm.predict(target.X)
    tyte_ = tsvm.transduced_labels

    iacc = (yte_ == target.y).mean()
    tacc = (tyte_ == target.y).mean()

    results.add(dataset='MDS', task=task, method='ISVM', fold=fold, acc=iacc)
    results.add(dataset='MDS', task=task, method='TSVM', fold=fold, acc=tacc)
results.pivot(grand_totals=True)


