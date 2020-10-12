from sklearn.model_selection import cross_val_score
from data.tasks import WebisCLS10_task_generator
from data.domain import unify_feat_space
from classification.svmlight import SVMlight
import os
from util.results import Result


dataset_home='../datasets/Webis-CLS-10'
svmlight_home='../../svm_light'

results = Result(['dataset', 'task', 'method', 'acc'])
for source, target, oracle, taskname in WebisCLS10_task_generator(os.path.abspath(dataset_home)):

    isvm = SVMlight(svmlightbase=svmlight_home, verbose=0)
    uacc = cross_val_score(isvm, target.X, target.y, cv=5).mean()
    results.add(dataset='Webis-CLS-10', task=taskname, method='Upper', acc=uacc)

    source, target = unify_feat_space(source, target)

    isvm = SVMlight(svmlightbase=svmlight_home, verbose=0).fit(source.X, source.y)
    tsvm = SVMlight(svmlightbase=svmlight_home, verbose=0, transduction=target.X).fit(source.X, source.y)

    iyte_ = isvm.predict(target.X)
    tyte_ = tsvm.transduced_labels

    iacc = (target.y == iyte_).mean()
    tacc = (target.y == tyte_).mean()

    results.add(dataset='Webis-CLS-10', task=taskname, method='ISVM', acc=iacc)
    results.add(dataset='Webis-CLS-10', task=taskname, method='TSVM', acc=tacc)

results.pivot(grand_totals=True)



