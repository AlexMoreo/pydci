from sklearn.model_selection import cross_val_score
from data.tasks import WebisCLS10_task_generator
from data.domain import pack_domains, unify_feat_space
from classification.svmlight import SVMlight
import os

dataset_home='../datasets/Webis-CLS-10'
svmlight_home='/home/moreo/svm_light'

results = []
up = []
for source, target, target_translations, oracle in WebisCLS10_task_generator(os.path.abspath(dataset_home), skip_translations=False):

    isvm = SVMlight(svmlightbase=svmlight_home, verbose=0)
    # uacc = cross_val_score(isvm, target_translations.X, target_translations.y, cv=5).mean() # <--
    uacc = cross_val_score(isvm, target.X, target.y, cv=5).mean()  # <--
    # isvm.fit(source.X, source.y)
    # uyte_ = isvm.predict(target_translations.X)
    # uacc = (target_translations.y == uyte_).mean()

    source, target = unify_feat_space(source, target)

    isvm = SVMlight(svmlightbase=svmlight_home, verbose=0)
    tsvm = SVMlight(svmlightbase=svmlight_home, verbose=0, transduction=target.X)

    isvm.fit(source.X, source.y)
    tsvm.fit(source.X, source.y)

    iyte_ = isvm.predict(target.X)
    tyte_=iyte_
    tyte_ = tsvm.transduced_labels

    iacc = (target.y == iyte_).mean()
    tacc = (target.y == tyte_).mean()

    print('{} {} : {:.3f} {:.3f} {:.3f}'.format(source.name(), target.name(), iacc, tacc, uacc))

    results.append((source.name()+'_'+target.name(), iacc, tacc, uacc))

print('Results Webis-cls-10')
for d,iacc,tacc,uacc in results:
    print('{}\t{:.3f}\t{:.3f}\t{:.3f}'.format(d,iacc,tacc,uacc))



