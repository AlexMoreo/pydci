from data.tasks import Topic_task_generator
from data.domain import *
from classification.svmlight import SVMlight
import numpy as np
from sklearn.model_selection import cross_val_score

twentynews_home='../datasets/20news'
sraa_home='../datasets/SRAA'
reuters_home= '../datasets/Reuters21578'
svmlight_home='/home/moreo/svm_light'


results = []
for source, target in Topic_task_generator(reuters_home=reuters_home, sraa_home=sraa_home, twenty_home=twentynews_home):
    continue

    source, target = unify_feat_space(source,target)

    isvm = SVMlight(svmlightbase=svmlight_home, verbose=0, transduction=None)
    tsvm = SVMlight(svmlightbase=svmlight_home, verbose=0, transduction=target.X)

    isvm.fit(source.X, source.y)
    tsvm.fit(source.X, source.y)

    yte_ = isvm.predict(target.X)
    tyte_ = tsvm.transduced_labels

    iacc = (yte_==target.y).mean()
    tacc = (tyte_==target.y).mean()
    upper = cross_val_score(isvm,target.X,target.y,cv=5).mean()

    print("{} {}: {:.3f} {:.3f} {:.3f}".format(source.domain,target.domain, iacc, tacc, upper))
    results.append((source.domain, target.domain, iacc, tacc, upper))

print('By Topic classification')
for s, t, iacc,tacc,upper in results:
    print("{} {}: {:.3f} {:.3f} {:.3f}".format(s,t,iacc,tacc,upper))

