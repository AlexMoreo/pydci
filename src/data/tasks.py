from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from data.fetch import *
import numpy as np
from sklearn.model_selection import StratifiedKFold
from data.domain import Domain, WordOracle


def as_domain(labeled_docs, labels, unlabeled_docs, issource, domain, translations=None, language='en', tokken_pattern=r"(?u)\b\w\w+\b", min_df=1, tfidf=True):
    """
    Represents raw documents as a Domain; a domain contains the tfidf weighted co-occurrence matrices of the labeled
    and unlabeled documents (with consistent Vocabulary).
    :param labeled_docs: the set of labeled documents
    :param labels: the labels of labeled_docs
    :param unlabeled_docs: the set of unlabeled documents
    :param issource: boolean, if True then the vocabulary is bounded to the labeled documents (the training set), if
    otherwise, then the vocabulary has to be bounded to that of the unlabeled set (which is expecteldy bigger) since
    we should assume the test set is only seen during evaluation. This is not true in a Transductive setting, but we
    force it to follow the same setting so as to allow for a fair evaluation.
    :param domain: the name of the domain (e.g., 'books'
    :param language: the language of the domain (e.g., 'french')
    :param tokken_pattern: the token pattern the sklearn vectorizer will use to split words
    :param min_df: the minimum frequency below which words will be filtered out from the vocabulary
    :return: an instance of Domain
    """
    if issource:
        counter = CountVectorizer(token_pattern=tokken_pattern, min_df=min_df)
        v = counter.fit(labeled_docs).vocabulary_
        if tfidf:
            vectorizer = TfidfVectorizer(sublinear_tf=True, token_pattern=tokken_pattern, vocabulary=v)
        else:
            vectorizer = CountVectorizer(token_pattern=tokken_pattern, vocabulary=v)
    else:
        if tfidf:
            vectorizer = TfidfVectorizer(sublinear_tf=True, token_pattern=tokken_pattern, min_df=min_df)
        else:
            vectorizer = CountVectorizer(token_pattern=tokken_pattern, min_df=min_df)
    U = vectorizer.fit_transform(unlabeled_docs)
    X = vectorizer.transform(labeled_docs)
    y = np.array(labels)
    V = vectorizer.vocabulary_
    domain = Domain(X, y, U, V, domain, language)
    if translations is not None:
        T = vectorizer.transform(translations)
        return domain, T
    else:
        return domain


def WebisCLS10_task_generator(dataset_home='../datasets/Webis-CLS-10', skip_translations=True, tfidf=True):
    """
    Generates the tasks for cross-lingual experiments in Webis-CLS-10 dataset
    :param dataset_home: the path where to store the dataset
    :return: yields tasks (source domain, target domain, and source-to-target oracle), in the typical order of
    appaerance of most papers
    """
    print('fetching Webis-CLS-10')
    documents, translations, dictionaries = fetch_Webis_cls_10(dataset_home=dataset_home, skip_translations=skip_translations, dopickle=True)

    patt = r"(?u)\b\w+\b" # japanese may contain words which are ony one symbol

    source_lan = 'en'
    taskno=0
    for target_lan in ['de', 'fr', 'jp']:
        for domain in ['books', 'dvd', 'music']:
            print('Loading Webis-CLS-10 task '+'{}{}-{}{}'.format(source_lan,domain,target_lan,domain).upper())

            tr_s_docs, tr_s_labels = list(zip(*documents[source_lan][domain]['train.processed']))
            unlabel_s_docs, _ = list(zip(*documents[source_lan][domain]['unlabeled.processed']))
            if not skip_translations:
                transl_t_docs, transl_t_labels = list(zip(*translations[target_lan][domain]['test.processed']))
                source, T = as_domain(tr_s_docs, tr_s_labels, unlabel_s_docs,
                                   issource=True, translations=transl_t_docs, domain=domain, language=source_lan,
                                   tokken_pattern=patt, min_df=1, tfidf=tfidf)
                Ty = np.array(transl_t_labels)
            else:
                source = as_domain(tr_s_docs, tr_s_labels, unlabel_s_docs,
                               issource=True, translations=None, domain=domain, language=source_lan,
                               tokken_pattern=patt, min_df=1, tfidf=tfidf)

            te_t_docs, te_t_labels = list(zip(*documents[target_lan][domain]['test.processed']))
            unlabel_t_docs, _ = list(zip(*documents[target_lan][domain]['unlabeled.processed']))
            target = as_domain(te_t_docs, te_t_labels, unlabel_t_docs,
                               issource=False, domain=domain, language=target_lan,
                               tokken_pattern=patt, min_df=3, tfidf=tfidf)

            oracle = WordOracle(dictionaries['{}_{}_dict.txt'.format(source_lan, target_lan)],
                                source_lan, target_lan, analyzer=CountVectorizer(token_pattern=patt).build_analyzer())

            print("source: X={} U={}".format(source.X.shape, source.U.shape))
            print("target: X={} U={}".format(target.X.shape, target.U.shape))

            taskname = '{}. {} {}'.format(taskno, source.name(), target.name())
            taskno+=1
            if skip_translations:
                yield source, target, oracle, taskname
            else:
                target_translations = Domain(T, Ty, None, source.V, domain, language='en')
                yield source, target, target_translations, oracle, taskname


def WebisCLS10_crossdomain_crosslingual_task_generator(dataset_home='../datasets/Webis-CLS-10', tfidf=True):
    """
    Generates the tasks for cross-lingual and cross-lingual (simultaneusly) experiments in Webis-CLS-10 dataset
    :param dataset_home: the path where to store the dataset
    :return: yields tasks (source domain, target domain, and source-to-target oracle).
    """
    print('fetching Webis-CLS-10')
    documents, translations, dictionaries = fetch_Webis_cls_10(dataset_home=dataset_home, skip_translations=True, dopickle=True)

    patt = r"(?u)\b\w+\b" # japanese may contain words which are ony one symbol

    source_lan = 'en'
    taskno=0
    for s_domain in ['books', 'dvd', 'music']:
        for target_lan in ['de', 'fr', 'jp']:
            for t_domain in ['books', 'dvd', 'music']:
                if s_domain == t_domain: continue

                print('Loading Webis-CLS-10 task '+'{}{}-{}{}'.format(source_lan,s_domain,target_lan,s_domain).upper())

                tr_s_docs, tr_s_labels = list(zip(*documents[source_lan][s_domain]['train.processed']))
                unlabel_s_docs, _ = list(zip(*documents[source_lan][s_domain]['unlabeled.processed']))
                source = as_domain(tr_s_docs, tr_s_labels, unlabel_s_docs,
                                   issource=True, translations=None, domain=s_domain, language=source_lan,
                                   tokken_pattern=patt, min_df=1, tfidf=tfidf)

                te_t_docs, te_t_labels = list(zip(*documents[target_lan][t_domain]['test.processed']))
                unlabel_t_docs, _ = list(zip(*documents[target_lan][t_domain]['unlabeled.processed']))
                target = as_domain(te_t_docs, te_t_labels, unlabel_t_docs,
                                   issource=False, domain=t_domain, language=target_lan,
                                   tokken_pattern=patt, min_df=3, tfidf=tfidf)

                oracle = WordOracle(dictionaries['{}_{}_dict.txt'.format(source_lan, target_lan)],
                                    source_lan, target_lan, analyzer=CountVectorizer(token_pattern=patt).build_analyzer())

                print("source: X={} U={}".format(source.X.shape, source.U.shape))
                print("target: X={} U={}".format(target.X.shape, target.U.shape))

                taskname = '{}. {} {}'.format(taskno, source.name(), target.name())
                taskno+=1
                yield source, target, oracle, taskname


def _extract_MDS_documents(documents, domain):
    pos_docs = [d for d, label in documents[domain]['positive.review']]
    neg_docs = [d for d, label in documents[domain]['negative.review']]
    unlabeled_docs = [d for d, label in documents[domain]['unlabeled.review']]
    labeled_docs = np.array(pos_docs + neg_docs)
    labels = np.array([1] * len(pos_docs) + [0] * len(neg_docs))
    return labeled_docs, labels, unlabeled_docs


def MDS_task_generator(dataset_home='../datasets/MDS', random_state=47, nfolds=5, tfidf=True):
    """
    Generates the tasks for cross-domain experiments in MDS dataset
    :param dataset_home: the path where to store the dataset
    :param random_state: allows to replicate random fold splits
    :return: yields tasks (source domain, target domain, number of fold) in the typical order of appaerance of most papers
    """
    print('fetching MDS')
    documents = fetch_MDS(dataset_home=dataset_home)

    domains = ['books', 'dvd', 'electronics', 'kitchen']
    for s_domain in domains:
        source_docs, source_labels, source_unlabel = _extract_MDS_documents(documents, s_domain)

        for t_domain in domains:
            if s_domain == t_domain: continue

            target_docs, target_labels, target_unlabel= _extract_MDS_documents(documents, t_domain)

            skf = StratifiedKFold(n_splits=nfolds, random_state=random_state, shuffle=True)
            for fold, (train_idx, test_idx) in enumerate(skf.split(source_docs, source_labels)):

                source = as_domain(source_docs[train_idx], source_labels[train_idx], source_unlabel,
                                   issource=True, domain=s_domain, min_df=3, tfidf=tfidf)
                target = as_domain(target_docs[test_idx], target_labels[test_idx], target_unlabel,
                                   issource=False, domain=t_domain, min_df=3, tfidf=tfidf)

                print("source: X={} U={}".format(source.X.shape, source.U.shape))
                print("target: X={} U={}".format(target.X.shape, target.U.shape))

                taskname = '{} {}'.format(source.domain, target.domain)

                yield source, target, fold, taskname


def UpperMDS_task_generator(dataset_home='../datasets/MDS', tfidf=True):

    print('fetching MDS')
    documents = fetch_MDS(dataset_home=dataset_home)

    domains = ['books', 'dvd', 'electronics', 'kitchen']
    for domain in domains:
        docs, labels, source_unlabel = _extract_MDS_documents(documents, domain)
        yield as_domain(docs, labels, source_unlabel, issource=True, domain=domain, min_df=3, tfidf=tfidf)


