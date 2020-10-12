import numpy as np
from scipy.sparse import csr_matrix
from time import time
from joblib import Parallel, delayed
from sklearn.preprocessing import normalize


class DCI:
    """
    Distributional Correspondence Indexing for domain adaptation.
    """

    prob_dcf = ['pmi', 'linear']
    vect_dcf = ['cosine']
    valid_dcf = prob_dcf + vect_dcf
    valid_post = ['normal', None]

    def __init__(self, dcf='cosine', post=None, n_jobs=-1, verbose=0, unify=False):
        """
        :param dcf: a distributional correspondence function name (e.g., 'cosine'), or a callable that, given two
                matrices F, P of shape (#features,#documents) and (#pivots,#documents) returns a matrix of
                shape (#features,#pivots). The dcf will be used to compute the distribucional correspondence between
                terms vectors
        :param post: post processing for projected documents ['normal' or None]; l2 is anyway computed, and optionally
                followed by a standardization if post='normal'
        :param n_jobs: number of parallel jobs; each domain is processed in parallel (though no more than two domains
                have been tested so far...)
        :param verbose: verbose level (the higher, the more verbose)
        :param unify: whether to unify the common terms (pivots are anyway unified)
        """
        if post not in self.valid_post:
            raise ValueError("unknown post processing function; valid ones are [%s]" % ', '.join(self.valid_post))

        if isinstance(dcf, str):
            if dcf not in self.valid_dcf:
                raise ValueError("unknown dcf; use any in [%s]" % ', '.join(self.valid_dcf))
            if dcf == 'cosine': self.dcf = cosine
            if dcf == 'pmi': self.dcf = pmi
            if dcf == 'linear': self.dcf = linear
        elif hasattr(dcf, '__call__'):
            self.dcf = dcf
        else:
            raise ValueError('param dcf should either be a valid dcf name in [%s] or a callable comparing two vectors')

        self.post = post
        self.domains = None
        self.dFP = None
        self.n_jobs = n_jobs
        self.unify = unify
        self.verbose = verbose


    def __print(self, msg, priority=0):
        if self.verbose >= priority:
            print(msg)


    def fit(self, dU, dP, dV=None):
        """
        :param dU: a dictionary of {domain:dsm_matrix}, where dsm is a document-by-term matrix representing the
                distributional semantic model for a specific domain (could came from any supervised or unsupervised matrix)
        :param dP: a dictionary {domain:pivot_indexes} where domain is a string representing each domain,
                and pivot_indexes is an array of length p, with p the number of pivots
        :param dV: a dictionary {domain:vocabulary} where domain is a string representing each domain,
                and vocabulary is an instance of Vocabulary for the domain (optional, useful if unification is True)
        :return: self
        """
        t_init = time()
        self.domains = list(dP.keys())
        assert len(np.unique([P.size for P in dP.values()])) == 1, "inconsistent number of pivots across domains"
        assert set(dU.keys()) == set(self.domains), "inconsistent domains in dU and dP"
        assert dV is None or set(dV.keys()) == set(self.domains), "inconsistent domains in dU and dP"
        self.dimensions = list(dP.values())[0].size

        self.dP = dP
        self.dV = dV

        # embed the feature space from each domain using the pivots of that domain
        transformations = Parallel(n_jobs=self.n_jobs)(delayed(dcf_dist)(dU[d].transpose(), dU[d][:,dP[d]].transpose(), self.dcf) for d in self.domains)
        self.dFP = {d: transformations[i] for i, d in enumerate(self.domains)}
        self.unification(self.dP) # pivots are always unified
        if self.unify: # unify (non-pivots) common-terms
            if dV is not None:
                dCommon = self.search_common_terms(dV)
                self.unification(dCommon)

        self.fit_time = time()-t_init
        return self


    def search_common_terms(self, dV):
        common_terms = list(set.intersection(*[vocabulary.term_set() for vocabulary in dV.values()]))
        dCommonIndexes = {d:np.array([dV[d].word2idx(idx) for idx in common_terms]) for d in self.domains}
        return dCommonIndexes


    # unifies all embeddings from dFP of the indexes in the dictionary dIndexes
    def unification(self, dIndexes):
        unified_matrix = np.array([self.dFP[d][dIndexes[d]] for d in self.domains]).mean(axis=0)
        for d in self.domains:
            self.dFP[d][dIndexes[d]] = unified_matrix


    # dX is a dictionary of {domain:dsm}, where dsm (distributional semantic model) is, e.g., a document-by-term csr_matrix
    def transform(self, dX):
        assert self.dFP is not None, 'transform method called before fit'
        assert set(dX.keys()).issubset(self.domains), 'domains in dX are not scope'
        t_init=time()
        domains = list(dX.keys())

        transformations = Parallel(n_jobs=self.n_jobs)(delayed(_dom_transform)(dX[d], self.dFP[d], self.post) for d in domains)
        transformations = {d: transformations[i] for i, d in enumerate(domains)}

        self.transform_time=time()-t_init
        return transformations


    def fit_transform(self, dU, dP, dX, dV=None):
        return self.fit(dU, dP, dV).transform(dX)

    def __str__(self):
        return "DCI({})".format(self.dcf.__name__)


def zscores(x):  #scipy.stats.zscores does not avoid division by 0, which can indeed occur
    std = np.clip(np.std(x, axis=0), 1e-5, None)
    mean = np.mean(x, axis=0)
    return (x - mean) / std


def dcf_dist(F, P, dcf):
    if not isinstance(F, csr_matrix): F = csr_matrix(F)
    if not isinstance(P, csr_matrix): P = csr_matrix(P)

    dists = dcf(F, P)

    # standardization
    dists = zscores(dists)

    # normalizing profiles to unit length
    normalize(dists, norm='l2', axis=1, copy=False)

    return dists


def _dom_transform(X, FP, post):
    _X = X.dot(FP)
    _X = normalize(_X, norm='l2', axis=1)
    if post == 'normal':
        _X = zscores(_X)
    return _X


# Distributional Correspondence Functions
# -----------------------------------------------------
def cosine(F, P):
    normalize(F, norm='l2', axis=1, copy=False)
    normalize(P, norm='l2', axis=1, copy=False)

    cos = F.dot(P.T)

    prevalences_F = (F > 0).mean(axis=1)
    prevalences_P = (P > 0).mean(axis=1)
    prev_factor = np.sqrt(np.outer(prevalences_F, prevalences_P.T))

    return cos - prev_factor


def pmi(F, P):
    nF,D = F.shape

    F=1*(F>0)
    P=1*(P>0)

    TP = F.dot(P.T).toarray().astype(np.float)
    FP = F.sum(axis=1) - TP
    FN = P.sum(axis=1).T - TP

    Ptp = TP / D
    Pfp = FP / D
    Pfn = FN / D

    denom = np.asarray(np.multiply(Pfp,Pfn))

    Ptp = np.divide(Ptp, denom, where=denom!=0)
    pmi = np.log2(Ptp, where=Ptp>0)
    pmi[np.isnan(pmi)]=0
    return pmi


def linear(F, P):
    _,D = F.shape

    F=1*(F>0)
    P=1*(P>0)

    TP = F.dot(P.T).toarray().astype(np.float)
    FP = np.asarray(F.sum(axis=1) - TP)
    FN = np.asarray(P.sum(axis=1).T - TP)
    TN = D-(TP+FP+FN)

    den1 = TP + FN
    TPR = np.divide(TP,den1,where=den1!=0)

    den2 = TN + FP
    TNR = np.divide(TN,den2,where=den2!=0)

    return TPR+TNR-1


