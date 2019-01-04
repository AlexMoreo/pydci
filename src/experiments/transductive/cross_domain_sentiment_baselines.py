from sklearn.model_selection import cross_val_score
from data.tasks import MDS_task_generator, UpperMDS_task_generator
from data.domain import *
from os.path import abspath
from classification.svmlight import SVMlight
import numpy as np

dataset_home='../datasets/MDS'
result_file = '../results/cross-domain_normal.csv'
svmlight_home='/home/moreo/svm_light'

nfolds=5
results = []
upper={}
for domain in UpperMDS_task_generator(abspath(dataset_home)):
    isvm = SVMlight(svmlightbase=svmlight_home, verbose=0, transduction=None)
    upper[domain.domain] = cross_val_score(isvm, domain.X, domain.y, cv=5).mean()
    print('{}: {:.3f}'.format(domain.domain, upper[domain.domain]))

cv_iaccs = []
cv_taccs = []
for source, target, fold in MDS_task_generator(abspath(dataset_home), nfolds=nfolds):

    source_name = source.domain
    target_name = target.domain
    source, target = unify_feat_space(source, target)

    isvm = SVMlight(svmlightbase=svmlight_home, verbose=0, transduction=None)
    tsvm = SVMlight(svmlightbase=svmlight_home, verbose=0, transduction=target.X)

    isvm.fit(source.X, source.y)
    tsvm.fit(source.X, source.y)

    yte_ = isvm.predict(target.X)
    tyte_ = tsvm.transduced_labels

    iacc = (yte_ == target.y).mean()
    tacc = (tyte_ == target.y).mean()

    cv_iaccs.append(iacc)
    cv_taccs.append(tacc)
    print('fold-{}: {} {} : {:.3f} {:.3f}'.format(fold, source.name(), target.name(), iacc, tacc))

    if fold == nfolds-1:
        iacc_ave = np.mean(cv_iaccs)
        tacc_ave = np.mean(cv_taccs)
        results.append((source.domain, target.domain, iacc_ave, tacc_ave, upper[target_name]))
        print('Average {} {} : {:.3f} {:.3f} {:.3f}'.format(source_name, target_name, iacc_ave, tacc_ave, upper[target_name]))
        cv_iaccs = []
        cv_taccs = []


print('MDS results')
for s,t,iacc,tacc,uacc in results:
    print('{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}'.format(s,t,iacc,tacc,uacc))
