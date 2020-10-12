from data.tasks import MDS_task_generator, WebisCLS10_task_generator
from os.path import abspath, exists
from time import time
from experiments.common import DCIinduction
from model.dci import DCI
from model.pivotselection import pivot_selection
import numpy as np
from util.results import Result

optimize = True

mds_home= '../datasets/MDS'
dataset_home='../datasets/Webis-CLS-10'

nfolds=5
outfile = './DCI.varpivot.dat'
if exists(outfile):
    rperf = Result.load(outfile, False)
else:
    rperf = Result(['dataset', 'task', 'method', 'fold', 'npivots', 'acc', 'dci_time', 'svm_time'])


pivot_range = [10,25,50,100,250,500,1000,1500,2000,2500,5000]

for source, target, fold, taskname in MDS_task_generator(abspath(mds_home), nfolds=nfolds):
    s_pivots, t_pivots = pivot_selection(max(pivot_range), source.X, source.y, source.U, target.U,
                                         source.V, target.V,
                                         phi=1, cross_consistency=True)

    for npivots in pivot_range:
        for dcf in ['cosine','linear']:
            dci = DCI(dcf=dcf, unify=False, post='normal')
            acc, dci_time, svm_time, _ = DCIinduction(source, target, s_pivots[:npivots], t_pivots[:npivots], dci, optimize=True)
            rperf.add(dataset='MDS', task=taskname, method=str(dci), fold=fold, npivots=npivots, acc=acc, dci_time=dci_time, svm_time=svm_time)
            rperf.pivot(index=['dataset', 'task','npivots'], values=['acc', 'dci_time', 'svm_time'])
            rperf.dump(outfile)

for source, target, oracle, taskname in WebisCLS10_task_generator(abspath(dataset_home)):
    s_pivots, t_pivots = pivot_selection(max(pivot_range), source.X, source.y, source.U, target.U,
                                         source.V, target.V,
                                         oracle=oracle, phi=30, cross_consistency=False)
    for npivots in pivot_range:
        t_pivots_sel = t_pivots[:npivots].astype(np.float)
        if not np.isnan(t_pivots_sel).any():
            for dcf in ['cosine','linear']:
                dci = DCI(dcf=dcf, unify=True, post='normal')
                acc, dci_time, svm_time, _ = DCIinduction(source, target, s_pivots[:npivots].astype(np.int), t_pivots_sel.astype(np.int), dci, optimize=True)
                rperf.add(dataset='Webis-CLS-10', task=taskname, method=str(dci), fold='0', npivots=npivots, acc=acc, dci_time=dci_time, svm_time=svm_time)
                rperf.pivot(index=['dataset', 'task', 'npivots'], values=['acc', 'dci_time', 'svm_time'])
                rperf.dump(outfile)
