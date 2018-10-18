from sklearn.svm import LinearSVC
from data.tasks import Topic_task_generator
from data.domain import *
from sklearn.model_selection import cross_val_score, GridSearchCV

from util.results import Result

twentynews_home='../../datasets/20news'
sraa_home='../../datasets/SRAA'
reuters_home= '../../datasets/Reuters21578'

results = Result(['dataset', 'task', 'method', 'acc'])
parameters = {'C': [10 ** i for i in range(-5, 5)]}
for source, target, task, dataset in Topic_task_generator(reuters_home=reuters_home, sraa_home=sraa_home, twenty_home=twentynews_home):
    source, target = unify_feat_space(source,target)

    svm = GridSearchCV(LinearSVC(), parameters, n_jobs=-1, verbose=1, cv=5)
    svm.fit(source.X, source.y)
    yte_ = svm.predict(target.X)
    acc = (yte_ == target.y).mean()
    results.add(dataset=dataset, task=task, method='Lower', acc=acc)

    svm = GridSearchCV(LinearSVC(), parameters, n_jobs=-1, verbose=1, cv=5)
    upper = cross_val_score(svm, target.X, target.y, cv=5).mean()
    results.add(dataset=dataset, task=task, method='Upper', acc=upper)

results.pivot(grand_totals=True)