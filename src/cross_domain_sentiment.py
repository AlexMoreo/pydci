from data.tasks import MDS_task_generatorsrc
from os.path import abspath
from experiments.common import DCIinduction, pivot_selection_timed
from model.dci import DCI
from util.results import Result


dcf= 'cosine'
npivots = 1000
optimize = True

nfolds=5
mds_home= '../datasets/MDS'

rperf = Result(['dataset', 'task', 'method', 'fold', 'acc','pivot_t', 'dci_t', 'svm_t', 'test_t'])
for source, target, fold, taskname in MDS_task_generator(abspath(mds_home), nfolds=nfolds):

    s_pivots, t_pivots, pivot_time = pivot_selection_timed(
        npivots, source.X, source.y, source.U, target.U, source.V, target.V, phi=1, show=min(10, npivots), cross=True
    )

    dci = DCI(dcf=dcf, unify=False, post='normal')
    acc, dci_time, svm_time, test_time = DCIinduction(source, target, s_pivots, t_pivots, dci, optimize=optimize)

    rperf.add(dataset='MDS', task=taskname, method=str(dci), fold=fold,
              acc=acc,
              pivot_t=pivot_time, dci_t=dci_time, svm_t=svm_time, test_t=test_time)

    rperf.dump(f'./DCI.{dcf}.m{npivots}.opt{optimize}.MDS.acc')
    rperf.pivot(grand_totals=True)
