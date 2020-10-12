from data.tasks_topics import Topic_task_generator
from model.dci import DCI
from experiments.common import DCIinduction, pivot_selection_timed
from util.results import Result


optimize = True
npivots = 1000
dcf='cosine'

twentynews_home='../datasets/20news'
sraa_home='../datasets/SRAA'
reuters_home= '../datasets/Reuters21578'

rperf = Result(['dataset', 'task', 'method', 'acc', 'pivot_t', 'dci_t', 'svm_t', 'test_t'])
for source, target, task, dataset in \
        Topic_task_generator(reuters_home=reuters_home, sraa_home=sraa_home, twenty_home=twentynews_home):

    s_pivots, t_pivots, pivot_time = pivot_selection_timed(
        npivots, source.X, source.y, source.U, target.U, source.V, target.V, phi=1, cross=True, show=min(10, npivots)
    )

    dci = DCI(dcf=dcf, unify=True, post='normal')
    acc, dci_time, svm_time, test_time = DCIinduction(source, target, s_pivots, t_pivots, dci, optimize=True)

    rperf.add(dataset=dataset, task=task, method=str(dci),
              acc=acc, pivot_t=pivot_time, dci_t=dci_time, svm_t=svm_time, test_t=test_time)

    rperf.dump(f'./DCI.{dcf}.m{npivots}.opt{optimize}.Topic.acc')
    rperf.pivot(grand_totals=True)


