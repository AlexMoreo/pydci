from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import LinearSVC
from data.tasks import WebisCLS10_task_generator, WebisCLS10_crossdomain_crosslingual_task_generator
from data.domain import unify_feat_space
import os

from util.results import Result

dataset_home='../../datasets/Webis-CLS-10'

results = Result(['dataset', 'task', 'method', 'acc'])

parameters = {'C': [10 ** i for i in range(-5, 5)]}
# for source, target, oracle, taskname in WebisCLS10_task_generator(os.path.abspath(dataset_home)):
for source, target, oracle, taskname in WebisCLS10_crossdomain_crosslingual_task_generator(os.path.abspath(dataset_home)):

    # upper
    svm = GridSearchCV(LinearSVC(), parameters, n_jobs=-1, verbose=1, cv=5)
    source, target = unify_feat_space(source, target)
    acc = cross_val_score(svm, target.X, target.y, cv=5).mean()
    results.add(dataset='Webis-CLS-10', task=taskname, method='Upper', acc=acc)

    # lower
    svm = GridSearchCV(LinearSVC(), parameters, n_jobs=-1, verbose=1, cv=5)
    svm.fit(source.X, source.y)
    yte_ = svm.predict(target.X)
    acc = (target.y == yte_).mean()
    results.add(dataset='Webis-CLS-10', task=taskname, method='Lower', acc=acc)

    results.pivot(grand_totals=True)