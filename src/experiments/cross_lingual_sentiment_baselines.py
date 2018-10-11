from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from data.tasks import WebisCLS10_task_generator
from data.domain import pack_domains, unify_feat_space
import os

from util.results import Result

dataset_home='../../datasets/Webis-CLS-10'

results = Result(['dataset', 'task', 'method', 'acc'])

for source, target, oracle, taskname in WebisCLS10_task_generator(os.path.abspath(dataset_home)):

    # upper
    svm = LinearSVC()
    source, target = unify_feat_space(source, target)
    acc = cross_val_score(svm, target.X, target.y, cv=5).mean()
    results.add(dataset='Webis-CLS-10', task=taskname, method='Upper', acc=acc)

    # lower
    svm = LinearSVC()
    svm.fit(source.X, source.y)
    yte_ = svm.predict(target.X)
    acc = (target.y == yte_).mean()
    acc = 0.5
    results.add(dataset='Webis-CLS-10', task=taskname, method='Lower', acc=acc)

results.pivot(grand_totals=True)