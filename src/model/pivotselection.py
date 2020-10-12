import numpy as np
from scipy.sparse import csc_matrix
from feature_selection.tsr_function import information_gain, ContTable


def pivot_selection(npivots, X, y, sU, tU, sV, tV,
                    oracle=None, tsr_function=information_gain, cross=True, phi=30, show=0, n_candidates =-1):
    X = csc_matrix(X)
    nD,nF = X.shape
    positives = y.sum()
    negatives = nD - positives

    # computes the 4-cell contingency tables for each feature
    TP = np.asarray((X[y==1]>0).sum(axis=0)).flatten()
    FN = positives - TP
    FP = np.asarray((X[y==0]>0).sum(axis=0)).flatten()
    TN = negatives - FP
    _4cell = [ContTable(tp=TP[i], tn=TN[i], fp=FP[i], fn=FN[i]) for i in range(nF)]

    # applies the tsr_function to the 4-cell counters
    feat_informativeness = np.array(list(map(tsr_function, _4cell)))

    if n_candidates==-1:
        n_candidates = npivots * 500
    candidates_idx_source = np.argsort(-feat_informativeness)[:n_candidates]
    feat_informativeness = feat_informativeness[candidates_idx_source]

    # counting feature frequencies cross consistency
    sU = csc_matrix(sU)
    tU = csc_matrix(tU)
    s_count = np.asarray((sU[:, candidates_idx_source] > 0).sum(axis=0)).flatten()
    s_prev = s_count / sU.shape[0]  # prevalence of terms in sU
    t_count = np.zeros_like(s_prev)
    for i, s_idx in enumerate(candidates_idx_source):
        t_idx = _s2t_idx(s_idx, sV, tV, oracle)
        if t_idx is not None:
            t_count[i] = (tU[:, t_idx] > 0).sum()
    t_prev = t_count / tU.shape[0]

    freq_threshold = (s_count > phi) * (t_count > phi)

    if cross:
        cross_consistency = np.minimum(s_prev, t_prev) / np.clip(np.maximum(s_prev, t_prev), 1e-5, None)
    else:
        cross_consistency = np.ones_like(s_prev)

    term_strength = feat_informativeness * cross_consistency * freq_threshold
    order_by_strength = np.argsort(-term_strength)
    term_strength = term_strength[order_by_strength]
    s_pivots = candidates_idx_source[order_by_strength[:npivots]]

    t_pivots = np.array([_s2t_idx(s_idx, sV, tV, oracle) for s_idx in s_pivots])

    #show top pivots
    show = min(show, npivots)
    for i in range(show):
        pivot_translation = (' (' + tV.idx2word(t_pivots[i]) + ')') if oracle else ''
        print(f'{i}: {sV.idx2word(s_pivots[i])}{pivot_translation} ({term_strength[i]:.4f})')
    if show > 0 and npivots > show:
        print('...')

    return s_pivots, t_pivots


def _s2t_idx(s_idx, sV, tV, oracle=None):
    """
    Translates an feature-index from the source domain to the feature-index of the target domain.
    If the source and target share the language, the vocabularies are the only data structures involved.
    If otherwise, the oracle is used to translate the word before retrieving the index.
    """
    t_idx = None
    word = sV.idx2word(s_idx)
    if oracle:
        word = oracle.source2target(word)
    if word:
        t_idx = tV.word2idx(word)
    return t_idx