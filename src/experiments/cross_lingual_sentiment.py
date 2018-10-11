from data.tasks import WebisCLS10_task_generator
from experiments.common import DCIclassify
from model.dci import DCI
from model.pivotselection import pivot_selection
import os
from time import time
from util.results import Result
import numpy as np

optimize = True
dcf='cosine'
npivots = 450
dataset_home='../datasets/Webis-CLS-10'

rperf = Result(['dataset', 'task', 'method', 'acc'])
rtime = Result(['dataset', 'task', 'method', 'pivot_t', 'dci_t', 'svm_t', 'test_t'])
for source, target, oracle, taskname in WebisCLS10_task_generator(os.path.abspath(dataset_home)):

    # pivot selection
    tinit = time()
    s_pivots, t_pivots = pivot_selection(npivots, source.X, source.y, source.U, target.U,
                                         source.V, target.V,
                                         oracle=oracle, phi=30, show=min(10, npivots), cross=False)
    pivot_time = time() - tinit
    print('pivot selection took {:.3f} seconds'.format(pivot_time))

    dci = DCI(dcf=dcf, unify=True, post='normal')
    acc, dci_time, svm_time, test_time = DCIclassify(source, target, s_pivots, t_pivots, dci, optimize=True)


    rperf.add(dataset='Webis-CLS-10', task=taskname, method=str(dci), acc=acc)
    rtime.add(dataset='Webis-CLS-10', task=taskname, method=str(dci),
              pivot_t=pivot_time, dci_t=dci_time, svm_t=svm_time, test_t=test_time)

rperf.pivot()
rtime.pivot(values=['pivot_t', 'dci_t', 'svm_t', 'test_t'], aggfunc=np.sum)

print('WITHOUT MAX CANDIDATES')



