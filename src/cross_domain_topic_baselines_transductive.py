from sklearn.model_selection import cross_val_score
from classification.svmlight import SVMlight
from data.domain import unify_feat_space
from data.tasks_topics import Topic_task_generator
from util.results import Result


twentynews_home='../datasets/20news'
sraa_home='../datasets/SRAA'
reuters_home= '../datasets/Reuters21578'
svmlight_home='../../svm_light'

results = Result(['dataset', 'task', 'method', 'acc'])
for source, target, task, dataset in \
        Topic_task_generator(reuters_home=reuters_home, sraa_home=sraa_home, twenty_home=twentynews_home):

    source, target = unify_feat_space(source,target)

    isvm = SVMlight(svmlightbase=svmlight_home, verbose=0, transduction=None).fit(source.X, source.y)
    tsvm = SVMlight(svmlightbase=svmlight_home, verbose=0, transduction=target.X).fit(source.X, source.y)

    yte_ = isvm.predict(target.X)
    tyte_ = tsvm.transduced_labels

    iacc = (yte_==target.y).mean()
    tacc = (tyte_==target.y).mean()
    results.add(dataset=dataset, task=task, method='ISVM', acc=iacc)
    results.add(dataset=dataset, task=task, method='TSVM', acc=tacc)

    upper = cross_val_score(isvm, target.X,target.y,cv=5).mean()
    results.add(dataset=dataset, task=task, method='Upper', acc=upper)

results.pivot(grand_totals=True)

