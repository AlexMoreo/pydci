from data.tasks_topics import Topic_task_generator
from experiments.common import DCItransduction, pivot_selection_timed
from model.dci import DCI
from util.results import Result


optimize = False
transductive = True
dcf='cosine'
npivots = 1000

twentynews_home='../datasets/20news'
sraa_home='../datasets/SRAA'
reuters_home= '../datasets/Reuters21578'
svmlight_home='../../svm_light'

methodname = ('T' if transductive else 'I') + f'DCI'

rperf = Result(['dataset', 'task', 'method', 'acc', 'pivot_t', 'dci_t', 'svm_t', 'test_t'])
for source, target, task, dataset in \
        Topic_task_generator(reuters_home=reuters_home, sraa_home=sraa_home, twenty_home=twentynews_home):

    s_pivots, t_pivots, pivot_time = pivot_selection_timed(
        npivots, source.X, source.y, source.U, target.U, source.V, target.V, phi=1, cross=True, show=min(npivots, 10)
    )

    dci = DCI(dcf=dcf, unify=True, post='normal')
    acc, dci_time, svm_time, test_time = DCItransduction(
        source, target, s_pivots, t_pivots, dci, svmlight_home, optimize=optimize, transductive=transductive
    )

    rperf.add(dataset=dataset, task=task, method=methodname,
              acc=acc, pivot_t=pivot_time, dci_t=dci_time, svm_t=svm_time, test_t=test_time)

    rperf.dump(f'./{methodname}.{dcf}.m{npivots}.opt{optimize}.Topic.acc')
    rperf.pivot(grand_totals=True)

