from data.tasks import MDS_task_generator
from os.path import abspath
from time import time
from experiments.common import DCIclassify
from model.dci import DCI
from model.pivotselection import pivot_selection
from util.results import Result


dcf= 'cosine'
npivots = 1000

optimize = True
mds_home= '../datasets/MDS'

nfolds=5
rperf = Result(['dataset', 'task', 'method', 'fold', 'acc','pivot_t', 'dci_t', 'svm_t', 'test_t'])

for source, target, fold, taskname in MDS_task_generator(abspath(mds_home), nfolds=nfolds):

    tinit = time()
    s_pivots, t_pivots = pivot_selection(npivots, source.X, source.y, source.U, target.U,
                                         source.V, target.V,
                                         phi=1, show=min(10, npivots), cross=True)
    pivot_time = time() - tinit
    print('pivot selection took {:.3f} seconds'.format(pivot_time))

    dci = DCI(dcf=dcf, unify=False, post='normal')
    acc, dci_time, svm_time, test_time = DCIclassify(source, target, s_pivots, t_pivots, dci, optimize=optimize)

    rperf.add(dataset='MDS', task=taskname, method=str(dci), fold=fold,
              acc=acc,
              pivot_t=pivot_time, dci_t=dci_time, svm_t=svm_time, test_t=test_time)

    rperf.dump('./DCI.{}.m{}.opt{}.MDS.acc'.format(dcf,npivots,optimize))

    rperf.pivot(grand_totals=True)
