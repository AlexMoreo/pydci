import pickle
from scipy.sparse import lil_matrix
import numpy as np

class Domain:
    """
    Defines a domain, composed by a labelled set and a unlabeled set. All sets share a common vocabulary.
    The domain is also characterized by its name and language.
    """

    def __init__(self, X, y, U, vocabulary, domain, language='en'):
        """
        :param X: the document collection
        :param y: the document labels
        :param U: the unlabeled collection
        :param vocabulary: the feature space of X and U
        :param domain: a descriptive name of the domain
        :param language: a descriptive name of the language
        """
        self.X = X
        self.y = y
        self.U = U
        self.V=vocabulary if isinstance(vocabulary, Vocabulary) else Vocabulary(vocabulary)
        self.domain = domain
        self.language = language

    def name(self):
        return '{}_{}'.format(self.language,self.domain)

    def dump(self, path):
        pickle.dump(self, open(path, 'wb'), pickle.HIGHEST_PROTOCOL)

    def show(self):
        print('domain: '+self.domain)
        print('language: ' + self.language)
        print('|V|={}'.format(len(self.V)))
        print('|X|={} (prev={})'.format(self.X.shape[0], self.y.mean()))
        print('|U|={}'.format(self.U.shape[0]))

    @classmethod
    def load(cls, path):
        domain = pickle.load(open(path, 'rb'))
        assert isinstance(domain, Domain), 'wrong pickle'
        return domain


class Vocabulary:
    """
    A bidirectional dictionary words->id and id->words
    """
    def __init__(self, word2idx_dict):
        self._word2idx = word2idx_dict
        self._idx2word = {idx:word for word, idx in word2idx_dict.items()}

    def word2idx(self, word):
        if word in self._word2idx:
            return self._word2idx[word]
        return None

    def idx2word(self, idx):
        if idx in self._idx2word:
            return self._idx2word[idx]
        return None

    def __len__(self):
        return len(self._word2idx)

    def term_set(self):
        return set(self._word2idx.keys())

    def index_list(self):
        return sorted(self._idx2word.keys())


class WordOracle:
    """
    An oracle that, given a source term returns the target translation, or viceversa.
    As defined by Prettenhofer, Peter, and Benno Stein. "Cross-language text classification using structural
    correspondence learning." Proceedings of the 48th annual meeting of the association for computational linguistics.
    Association for Computational Linguistics, 2010.
    """
    def __init__(self, dictionary, source, target, analyzer=None):
        self.source = source
        self.target = target
        self.s2t_dict = {_preproc(analyzer, s) : _preproc(analyzer, t) for s, t in dictionary.items()} if analyzer else dictionary
        self.t2s_dict = {v:k for k,v in dictionary.items()}

    def source2target(self, word):
        if word in self.s2t_dict.keys():
            return self.s2t_dict[word]
        return None

    def target2source(self, word):
        if word in self.t2s_dict.keys():
            return self.t2s_dict[word]
        return None


def _preproc(analyzer, str):
    return analyzer(str)[0] if analyzer(str) else 'null__'


def pack_domains(source, target, pivots_source, pivots_target):
    dX = {source.name(): source.X, target.name(): target.X}
    dU = {source.name(): source.U, target.name(): target.U}
    dP = {source.name(): pivots_source, target.name(): pivots_target}
    dV = {source.name(): source.V, target.name(): target.V}
    return dX, dU, dP, dV


def unify_feat_space(source, target):
    """
    Given a source and a target domain, returns two new versions of them in which the feature spaces are common, by
    trivially juxtapossing the two vocabularies
    :param source: the source domain
    :param target: the target domain
    :return: a new version of the source and the target domains where the feature space is common
    """
    word_set = source.V.term_set().union(target.V.term_set())
    word2idx = {w:i for i,w in enumerate(word_set)}
    Vshared = Vocabulary(word2idx)

    def reindexDomain(domain, sharedV):
        V = domain.V
        nD=domain.X.shape[0]
        nF=len(sharedV)
        newX = lil_matrix((nD,nF))
        domainIndexes = np.array(V.index_list())
        sharedIndexes = np.array([sharedV.word2idx(w) for w in [V.idx2word(i) for i in domainIndexes]])
        newX[:,sharedIndexes]=domain.X[:,domainIndexes]
        return Domain(newX.tocsr(),domain.y,None,sharedV,domain.domain+'_shared',domain.language)

    return reindexDomain(source, Vshared), reindexDomain(target, Vshared)