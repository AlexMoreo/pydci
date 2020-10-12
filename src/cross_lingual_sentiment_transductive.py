from data.tasks import WebisCLS10_task_generator
from experiments.common import pivot_selection_timed, DCItransduction
from model.dci import DCI
import os
from util.results import Result


optimize = False
transductive = True
dcf='cosine'
npivots = 450
svmlight_home='../../svm_light'
dataset_home='../datasets/Webis-CLS-10'

methodname = ('T' if transductive else 'I') + f'DCI'

rperf = Result(['dataset', 'task', 'method', 'acc', 'pivot_t', 'dci_t', 'svm_t', 'test_t'])
for source, target, oracle, taskname in WebisCLS10_task_generator(os.path.abspath(dataset_home)):

    s_pivots, t_pivots, pivot_time = pivot_selection_timed(
        npivots, source.X, source.y, source.U, target.U, source.V, target.V,
        oracle=oracle, phi=30, show=min(20, npivots), cross=False
    )

    dci = DCI(dcf=dcf, unify=True, post='normal')
    acc, dci_time, svm_time, test_time = DCItransduction(
        source, target, s_pivots, t_pivots, dci, svmlight_home, optimize=optimize, transductive=transductive
    )

    rperf.add(dataset='Webis-CLS-10', task=taskname, method=methodname,
              acc=acc,
              pivot_t=pivot_time, dci_t=dci_time, svm_t=svm_time, test_t=test_time)

    rperf.dump(f'./{methodname}.{dcf}.m{npivots}.opt{optimize}.WebisCLS10.acc')
    rperf.pivot(grand_totals=True)




