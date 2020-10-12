from data.tasks import WebisCLS10_crossdomain_crosslingual_task_generator
from experiments.common import DCIinduction, pivot_selection_timed
from model.dci import DCI
import os
from util.results import Result


dcf = 'cosine'
npivots = 450
optimize = True

dataset_home = '../datasets/Webis-CLS-10'

rperf = Result(['dataset', 'task', 'method', 'acc', 'pivot_t', 'dci_t', 'svm_t', 'test_t'])
for source, target, oracle, taskname in WebisCLS10_crossdomain_crosslingual_task_generator(os.path.abspath(dataset_home)):

    s_pivots, t_pivots, pivot_time = pivot_selection_timed(
        npivots, source.X, source.y, source.U, target.U, source.V, target.V,
        oracle=oracle, phi=30, show=min(10, npivots), cross=True
    )

    dci = DCI(dcf=dcf, unify=True, post='normal')
    acc, dci_time, svm_time, test_time = DCIinduction(source, target, s_pivots, t_pivots, dci, optimize=optimize)

    rperf.add(dataset='Webis-CLS-10', task=taskname, method=str(dci), acc=acc,
              pivot_t=pivot_time, dci_t=dci_time, svm_t=svm_time, test_t=test_time)

    rperf.dump(f'./DCI.{dcf}.m{npivots}.opt{optimize}.WebisCLS10.crossdom_crosslin.acc')
    rperf.pivot(grand_totals=True)





