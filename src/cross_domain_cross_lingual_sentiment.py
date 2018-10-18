from data.tasks import WebisCLS10_task_generator, WebisCLS10_crossdomain_crosslingual_task_generator
from experiments.common import DCIclassify
from model.dci import DCI
from model.pivotselection import pivot_selection
import os
from time import time
from util.results import Result

optimize = True
dcf='linear'
npivots = 450
norm='normal'
dataset_home='../datasets/Webis-CLS-10'

rperf = Result(['dataset', 'task', 'method', 'acc', 'pivot_t', 'dci_t', 'svm_t', 'test_t'])
for source, target, oracle, taskname in WebisCLS10_crossdomain_crosslingual_task_generator(os.path.abspath(dataset_home)):

    # pivot selection
    tinit = time()
    s_pivots, t_pivots = pivot_selection(npivots, source.X, source.y, source.U, target.U,
                                         source.V, target.V,
                                         oracle=oracle, phi=30, show=min(10, npivots), cross=True)
    pivot_time = time() - tinit
    print('pivot selection took {:.3f} seconds'.format(pivot_time))

    dci = DCI(dcf=dcf, unify=True, post=norm)
    acc, dci_time, svm_time, test_time = DCIclassify(source, target, s_pivots, t_pivots, dci, optimize=optimize)


    rperf.add(dataset='Webis-CLS-10', task=taskname, method=str(dci),
              acc=acc,
              pivot_t=pivot_time, dci_t=dci_time, svm_t=svm_time, test_t=test_time)

    rperf.dump('./DCI.{}.m{}.opt{}.norm{}.WebisCLS10.crossdom_crosslin.acc'.format(dcf, npivots, optimize, norm))
rperf.pivot()





