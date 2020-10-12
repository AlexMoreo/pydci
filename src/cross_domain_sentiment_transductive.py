from data.tasks import MDS_task_generator
from os.path import abspath
from experiments.common import pivot_selection_timed, DCItransduction
from model.dci import DCI
from util.results import Result


dcf= 'cosine'
npivots = 1000
optimize = False
transductive = False

nfolds=5
mds_home= '../datasets/MDS'
svmlight_home='../../svm_light'

methodname = ('T' if transductive else 'I') + f'DCI'

rperf = Result(['dataset', 'task', 'method', 'fold', 'acc','pivot_t', 'dci_t', 'svm_t', 'test_t'])
for source, target, fold, taskname in MDS_task_generator(abspath(mds_home), nfolds=nfolds):

    s_pivots, t_pivots, pivot_time = pivot_selection_timed(
        npivots, source.X, source.y, source.U, target.U, source.V, target.V, phi=1, show=min(20, npivots), cross=True
    )

    dci = DCI(dcf=dcf, unify=False, post='normal')
    acc, dci_time, svm_time, test_time = DCItransduction(
        source, target, s_pivots, t_pivots, dci, svmlight_home, optimize=optimize, transductive=transductive
    )

    rperf.add(dataset='MDS', task=taskname, method=methodname, fold=fold,
              acc=acc,
              pivot_t=pivot_time, dci_t=dci_time, svm_t=svm_time, test_t=test_time)

    rperf.dump(f'./{methodname}.{dcf}.m{npivots}.opt{optimize}.MDS.acc')
    rperf.pivot(grand_totals=True)


