import tarfile
import pickle
import zipfile
from sklearn.datasets import load_files
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_header, strip_newsgroup_footer, strip_newsgroup_quoting
from util.file import *
from scipy.io import arff
from scipy.sparse import csr_matrix
import numpy as np
import sklearn

def _proc_review(doc):
    parts = doc.split(' ')
    label = parts[-1].replace('#label#:', '').strip()
    assert label in ['positive','negative'], 'error parsing label {}'.format(label)
    label = 1 if label == 'positive' else 0
    repeat_word = lambda word, num: ' '.join([word] * int(num))
    text = ' '.join([repeat_word(*x.split(':')) for x in parts[:-1]])
    return text, label

def fetch_MDS(DOWNLOAD_URL = 'https://www.cs.jhu.edu/~mdredze/datasets/sentiment/processed_acl.tar.gz',
              dataset_home = '../datasets/MDS'):
    """
    Fetchs the processed version of the  Multi-Domain Sentiment Dataset (version 2.0) for cross-domain adaptation defined in:
    John Blitzer, Mark Dredze, Fernando Pereira.
    Biographies, Bollywood, Boom-boxes and Blenders: Domain Adaptation for Sentiment Classification.
    Association of Computational Linguistics (ACL), 2007.
    """
    dataset_path = join(dataset_home, 'processed_acl.tar.gz')
    create_if_not_exist(dataset_home)

    if not exists(dataset_path):
        print("downloading multidomain dataset (once and for all) into %s" % dataset_path)
        download_file(DOWNLOAD_URL, dataset_path)
        print("untarring dataset...")
        tarfile.open(dataset_path, 'r:gz').extractall(dataset_home)

    dataset_path = dataset_path.replace('.tar.gz','')

    documents = dict()
    for domain in list_dirs(dataset_path):
        documents[domain] = dict()
        for file in ['positive.review', 'negative.review', 'unlabeled.review']:
            documents[domain][file] = []
            for doc in open(join(dataset_path, domain, file), 'rt'):
                text, label = _proc_review(doc)
                documents[domain][file].append((text,label))
            print('{} documents read for domain {} in file {}'.format(len(documents[domain][file]), domain, file))

    return documents


def fetch_Webis_cls_10(DOWNLOAD_URL = 'https://zenodo.org/record/3251672/files/cls-acl10-processed.tar.gz?download=1',
                       #DOWNLOAD_URL = 'http://www.uni-weimar.de/medien/webis/corpora/corpus-webis-cls-10/cls-acl10-processed.tar.gz',
                       dataset_home = '../datasets/Webis-CLS-10',
                       max_documents=50000,
                       languages=['de','en','fr','jp'],
                       domains=None,
                       skip_translations=False,
                       dopickle=False):
    """
    Fetchs the processed version of the Webis-CLS-10 dataset for cross-lingual adaptation defined in:
    Prettenhofer, Peter, and Benno Stein.
    "Cross-language text classification using structural correspondence learning."
    Proceedings of the 48th annual meeting of the association for computational linguistics.
    Association for Computational Linguistics, 2010.
    """
    picklepath = join(dataset_home, 'webiscls10_processed.pkl')
    if dopickle and exists(picklepath):
        print('...loading pickle from {}'.format(picklepath))
        return pickle.load(open(picklepath, 'rb'))

    dataset_path = join(dataset_home, 'cls-acl10-processed.tar.gz')
    create_if_not_exist(dataset_home)

    if not exists(dataset_path):
        print("downloading Webis-CLS1-0 dataset (once and for all) into %s" % dataset_path)
        download_file(DOWNLOAD_URL, dataset_path)
        print("untarring dataset...")
        tarfile.open(dataset_path, 'r:gz').extractall(dataset_home)

    dataset_path = dataset_path.replace('.tar.gz','')

    documents = dict()
    for language in languages:
        documents[language] = dict()
        domain_list = domains if domains is not None else list_dirs(join(dataset_path,language))
        for domain in domain_list:
            documents[language][domain] = dict()
            for file in ['train.processed', 'test.processed', 'unlabeled.processed']:
                documents[language][domain][file] = []
                for doc in open(join(dataset_path, language, domain, file), 'rt'):
                    text, label = _proc_review(doc)
                    documents[language][domain][file].append((text,label))
                    if max_documents is not None and len(documents[language][domain][file]) >= max_documents:
                        break
                print('{} documents read for language {}, domain {}, in file {}'.format(
                    len(documents[language][domain][file]), language, domain, file))

    translations = dict()
    if not skip_translations:
        for language in ['de','fr','jp']:
            translations[language] = dict()
            for domain in list_dirs(join(dataset_path,language)):
                translations[language][domain] = dict()
                for file in ['test.processed']:
                    translations[language][domain][file] = []
                    for doc in open(join(dataset_path, language, domain, 'trans', 'en', domain, file), 'rt'):
                        text, label = _proc_review(doc)
                        translations[language][domain][file].append((text,label))
                    print('{} translations read for language {}, domain {}, in file {}'.format(
                        len(translations[language][domain][file]), language, domain, file))

    dictionaries = dict()
    split_t = lambda x: x.strip().replace(' ','').split('\t')
    for d in list_files(join(dataset_path,'dict')):
        dictionaries[d] = dict(map(split_t, open(join(dataset_path,'dict',d))))

    if dopickle:
        print('...pickling the dataset into {} to speed-up next calls'.format(picklepath))
        pickle.dump((documents, translations, dictionaries), open(picklepath, 'wb'), pickle.HIGHEST_PROTOCOL)

    return documents, translations, dictionaries



def fetch_reuters21579(DOWNLOAD_URL = 'http://www.cse.ust.hk/TL/dataset/Reuters.zip',
                       dataset_home='../datasets/Reuters21578', dopickle=False):
    """
    Fetchs a version of Reuters21578 for cross-domain adaptation, as defined in:
    Dai, W., Xue, G. R., Yang, Q., & Yu, Y. (2007, August).
    Co-clustering based classification for out-of-domain documents.
    In Proceedings of the 13th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 210-219). ACM.
    """
    picklepath = join(dataset_home, 'reuters.pkl')
    if dopickle and exists(picklepath):
        print('...loading pickle from {}'.format(picklepath))
        return pickle.load(open(picklepath, 'rb'))

    dataset_path = join(dataset_home, 'Reuters.zip')

    if not exists(dataset_path):
        create_if_not_exist(dataset_home)
        print("downloading Reuters dataset (once and for all) into %s" % dataset_path)
        download_file(DOWNLOAD_URL, dataset_path)
        print("unzipping dataset...")
        zip_ref = zipfile.ZipFile(dataset_path, 'r')
        zip_ref.extractall(dataset_home)
        zip_ref.close()

    corpus = {}
    for file in list_files(dataset_home):
        if not file.endswith('.arff'): continue
        task, domain, _ = file.split('.')
        data, meta = arff.loadarff(join(dataset_home, file))
        data = np.array([[int(x) for x in row] for row in data])
        X = csr_matrix(data[:, :-1])
        y = data[:, -1]
        if task not in corpus: corpus[task] = {}
        corpus[task][domain] = (X, y)
        print('loaded {} with shape {} and prevalence {}'.format(file, X.shape, y.mean()))

    if dopickle:
        print('...pickling the dataset into {} to speed-up next calls'.format(picklepath))
        pickle.dump(corpus, open(picklepath, 'wb'), pickle.HIGHEST_PROTOCOL)

    return corpus


def _index_by_label(news, min_words=1):
    corpus = {label:[] for label in news.target_names}
    removed=0
    for i,doc in enumerate(news.data):
        if len(doc.strip().split())>=min_words:
            label_id = news.target[i]
            label = news.target_names[label_id]
            corpus[label].append(doc)
        else: removed+=1
    print('removed {} documents'.format(removed))
    return corpus


def fetch_20newsgroups(dataset_home='../datasets/20news', dopickle=False):
    """
    Fetchs a version of 20Newsgroups for cross-domain adaptation, as defined in:
    Dai, W., Xue, G. R., Yang, Q., & Yu, Y. (2007, August).
    Co-clustering based classification for out-of-domain documents.
    In Proceedings of the 13th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 210-219). ACM.
    """
    picklepath = join(dataset_home, '20news.pkl')
    if dopickle and exists(picklepath):
        print('...loading pickle from {}'.format(picklepath))
        return pickle.load(open(picklepath, 'rb'))

    news = sklearn.datasets.fetch_20newsgroups(data_home=dataset_home, subset='all')
    news = _index_by_label(news)

    if dopickle:
        print('...pickling the dataset into {} to speed-up next calls'.format(picklepath))
        pickle.dump(news, open(picklepath, 'wb'), pickle.HIGHEST_PROTOCOL)

    return news


def fetch_sraa(DOWNLOAD_URL = 'http://people.cs.umass.edu/~mccallum/data/sraa.tar.gz',
                       dataset_home='../datasets/SRAA', dopickle=False):
    """
    Fetchs a version of Reuters21578 for cross-domain adaptation, as defined in:
    Dai, W., Xue, G. R., Yang, Q., & Yu, Y. (2007, August).
    Co-clustering based classification for out-of-domain documents.
    In Proceedings of the 13th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 210-219). ACM.
    """
    picklepath = join(dataset_home, 'sraa.pkl')
    if dopickle and exists(picklepath):
        print('...loading pickle from {}'.format(picklepath))
        return pickle.load(open(picklepath, 'rb'))

    dataset_path = join(dataset_home, 'sraa.tar.gz')

    if not exists(dataset_path):
        create_if_not_exist(dataset_home)
        print("downloading SRAA dataset (once and for all) into %s" % dataset_path)
        download_file(DOWNLOAD_URL, dataset_path)
        print("untarring dataset...")
        tarfile.open(dataset_path, 'r:gz').extractall(dataset_home)

    sraa = load_files(join(dataset_home,'sraa'), encoding='latin1')
    remove = ('headers', 'footers')#, 'quotes')
    if 'headers' in remove:
        sraa.data = [strip_newsgroup_header(text) for text in sraa.data]
    if 'footers' in remove:
        sraa.data = [strip_newsgroup_footer(text) for text in sraa.data]
    if 'quotes' in remove:
        sraa.data = [strip_newsgroup_quoting(text) for text in sraa.data]

    sraa = _index_by_label(sraa, min_words=10)

    if dopickle:
        print('...pickling the dataset into {} to speed-up next calls'.format(picklepath))
        pickle.dump(sraa, open(picklepath, 'wb'), pickle.HIGHEST_PROTOCOL)

    return sraa
