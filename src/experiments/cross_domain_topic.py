from data.tasks import Topic_task_generator
from time import time
from model.dci import DCI
from model.pivotselection import pivot_selection
from experiments.common import DCIclassify
from util.results import Result
import numpy as np

optimize = True
npivots = 1000
dcf='linear'

twentynews_home='../datasets/20news'
sraa_home='../datasets/SRAA'
reuters_home= '../datasets/Reuters21578'

rperf = Result(['dataset', 'task', 'method', 'acc'])
rtime = Result(['dataset', 'task', 'method', 'pivot_t', 'dci_t', 'svm_t', 'test_t'])
for source, target, task, dataset in Topic_task_generator(reuters_home=reuters_home, sraa_home=sraa_home, twenty_home=twentynews_home):

    tinit = time()
    s_pivots, t_pivots = pivot_selection(npivots, source.X, source.y, source.U, target.U,
                                         source.V, target.V,
                                         phi=1, cross=True, show=min(10, npivots))
    pivot_time = time() - tinit
    print('pivot selection took {:.3f} seconds'.format(pivot_time))

    dci = DCI(dcf=dcf, unify=True, post='normal')
    acc, dci_time, svm_time, test_time = DCIclassify(source, target, s_pivots, t_pivots, dci, optimize=True)

    # pivot_time, acc, dci_time, svm_time, test_time = 0.1, 0.2, 0.3, 0.4, 0.5
    rperf.add(dataset=dataset, task=task, method=str(dci), acc=acc)
    rtime.add(dataset=dataset, task=task, method=str(dci),
              pivot_t=pivot_time, dci_t=dci_time, svm_t=svm_time, test_t=test_time)

rperf.pivot()
rtime.pivot(values=['pivot_t', 'dci_t', 'svm_t', 'test_t'], aggfunc=np.sum)

