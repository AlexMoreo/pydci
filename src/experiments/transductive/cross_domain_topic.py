from data.tasks import Topic_task_generator
from time import time
from experiments.transductive.common import DCItransduction
from model.dci import DCI
from model.pivotselection import pivot_selection


optimize = True
transductive = False
npivots = 1000
dcf='linear'

twentynews_home='../datasets/20news'
sraa_home='../datasets/SRAA'
reuters_home= '../datasets/Reuters21578'
svmlight_home='/home/moreo/svm_light'

results = []

for source, target in Topic_task_generator(reuters_home=reuters_home, sraa_home=sraa_home, twenty_home=twentynews_home):

    print('Pivot selection')
    tinit = time()
    s_pivots, t_pivots = pivot_selection(npivots, source.X, source.y, source.U, target.U,
                                         source.V, target.V,
                                         phi=1, cross_consistency=True, show=min(npivots, 10))
    pivot_time = time() - tinit
    print('pivot selection took {} seconds'.format(pivot_time))

    dci = DCI(dcf=dcf, unify=True, post='normal')
    acc, dci_time, svm_time, test_time = DCItransduction(source, target, s_pivots, t_pivots, dci, svmlight_home, optimize=False, transductive=True)

    results.append((source.name(), target.name(), acc))
    print('{} {} : {:.3f}'.format(source.name(), target.name(), acc))

print('Transductive {}, optimization {}, pivots={}, dcf={}'.format(transductive, optimize, npivots, dcf))
for s, t, acc in results:
    print('{}\t{}\t{:.3f}'.format(s, t, acc))
