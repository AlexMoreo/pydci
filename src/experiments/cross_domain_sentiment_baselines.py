from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import LinearSVC

from data.domain import unify_feat_space
from data.tasks import MDS_task_generator, UpperMDS_task_generator
from os.path import abspath
from util.results import Result

dataset_home='../../datasets/MDS'

results = Result(['dataset', 'task', 'method', 'fold', 'acc'])
nfolds=5

upper={}
parameters = {'C': [10 ** i for i in range(-5, 5)]}

for domain in UpperMDS_task_generator(abspath(dataset_home)):
    svm = GridSearchCV(LinearSVC(), parameters, n_jobs=-1, verbose=1, cv=5)
    upper[domain.domain] = cross_val_score(svm, domain.X, domain.y, cv=5).mean()


for source, target, fold, taskname in MDS_task_generator(abspath(dataset_home), nfolds=nfolds):
    svm = GridSearchCV(LinearSVC(), parameters, n_jobs=-1, verbose=1, cv=5)
    target_domain_name = target.domain
    source, target = unify_feat_space(source, target)
    svm.fit(source.X, source.y)
    yte_ = svm.predict(target.X)
    acc = (yte_ == target.y).mean()

    results.add(dataset='MDS', task=taskname, method='Lower', fold=fold, acc=acc)
    results.add(dataset='MDS', task=taskname, method='Upper', fold=fold, acc=upper[target_domain_name])

results.pivot(grand_totals=True)