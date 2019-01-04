from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from data.fetch import *
import numpy as np
from data.domain import Domain, WordOracle, Vocabulary, unify_feat_space
from itertools import chain


def Reuters_task_generator(dataset_home='../datasets/Reuters'):

    print('fetching Reuters')
    reuters = fetch_reuters21579(dataset_home=dataset_home, dopickle=True)

    for order,task in enumerate(['OrgsPlaces', 'OrgsPeople', 'PeoplePlaces']):
        Xs, ys = reuters[task]['src']
        Xt, yt = reuters[task]['tar']
        assert Xs.shape[1]==Xt.shape[1], 'wrong number of columns'

        tfidf = TfidfTransformer(sublinear_tf=True)
        Xs = tfidf.fit_transform(Xs)
        Xt = tfidf.transform(Xt)

        fake_vocab = {'f%d'%i:i for i in range(Xs.shape[1])}
        source = Domain(Xs, ys, Xs, fake_vocab, task+'_source')
        target = Domain(Xt, yt, Xt, fake_vocab, task+'_target')

        print('X.shape={}, y-prevalence={:.3f}'.format(source.X.shape, source.y.mean()))
        print('X.shape={}, y-prevalence={:.3f}'.format(target.X.shape, target.y.mean()))

        yield source, target, '{}. {}'.format(order,task)


def _domain_from_usenet(news, positive, negative, domain_name, max_documents=None):
    pos_docs = list(chain(*[news[l] for l in positive]))
    neg_docs = list(chain(*[news[l] for l in negative]))

    if max_documents is not None:
        pos_docs = pos_docs[:int(max_documents/2)]
        neg_docs = neg_docs[:max_documents-len(pos_docs)]

    all_docs = pos_docs+neg_docs
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=3, strip_accents='unicode')
    X = tfidf.fit_transform(all_docs)
    y = np.array([1] * len(pos_docs) + [0] * len(neg_docs))
    V = tfidf.vocabulary_
    print('X.shape={}, y-prevalence={:.3f}'.format(X.shape, y.mean()))
    return Domain(X, y, X, V, domain_name)


def TwentyNews_task_generator(dataset_home='../datasets/20news'):
    print('fetching 20 Newsgroups')
    news = fetch_20newsgroups(dataset_home=dataset_home, dopickle=True)

    print('comp vs sci')
    source = _domain_from_usenet(news, domain_name='comp_vs_sci',
                                 positive=['comp.graphics', 'comp.os.ms-windows.misc'],
                                 negative=['sci.crypt', 'sci.electronics'])
    target = _domain_from_usenet(news, domain_name='comp_vs_sci_target',
                                 positive=['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x'],
                                 negative=['sci.med', 'sci.space'])
    yield source, target, '0. comp vs sci'

    print('rec vs talk')
    source = _domain_from_usenet(news, domain_name='rec_vs_talk',
                                 positive=['rec.autos', 'rec.motorcycles'],
                                 negative=['talk.politics.guns','talk.politics.misc'])
    target = _domain_from_usenet(news, domain_name='rec_vs_talk_target',
                                 positive=['rec.sport.baseball', 'rec.sport.hockey'],
                                 negative=['talk.politics.mideast','talk.religion.misc'])
    yield source, target, '1. rec vs talk'

    print('rec vs sci')
    source = _domain_from_usenet(news, domain_name='rec_vs_sci',
                                 positive=['rec.autos', 'rec.sport.baseball'],
                                 negative=['sci.med','sci.space'])
    target = _domain_from_usenet(news, domain_name='rec_vs_sci_target',
                                 positive=['rec.motorcycles', 'rec.sport.hockey'],
                                 negative=['sci.crypt','sci.electronics'])
    yield source, target, '2. rec vs sci'

    print('sci vs talk')
    source = _domain_from_usenet(news, domain_name='sci_vs_talk',
                                 positive=['sci.electronics', 'sci.med'],
                                 negative=['talk.politics.misc','talk.religion.misc'])
    target = _domain_from_usenet(news, domain_name='sci_vs_talk_target',
                                 positive=['sci.crypt', 'sci.space'],
                                 negative=['talk.politics.guns','talk.politics.mideast'])
    yield source, target, '3. sci vs talk'

    print('comp vs rec')
    source = _domain_from_usenet(news, domain_name='comp_vs_rec',
                                 positive=['comp.graphics', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware'],
                                 negative=['rec.motorcycles','rec.sport.hockey'])
    target = _domain_from_usenet(news, domain_name='comp_vs_rec_target',
                                 positive=['comp.os.ms-windows.misc', 'comp.windows.x'],
                                 negative=['rec.autos','rec.sport.baseball'])
    yield source, target, '4. comp vs rec'

    print('comp vs talk')
    source = _domain_from_usenet(news, domain_name='comp_vs_talk',
                                 positive=['comp.graphics', 'comp.sys.mac.hardware', 'comp.windows.x'],
                                 negative=['talk.politics.mideast','talk.religion.misc'])
    target = _domain_from_usenet(news, domain_name='comp_vs_talk_target',
                                 positive=['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware'],
                                 negative=['talk.politics.guns','talk.politics.misc'])
    yield source, target, '5. comp vs talk'



def SRAA_task_generator(dataset_home='../datasets/SRAA'):

    print('fetching SRAA')
    sraa = fetch_sraa(dataset_home=dataset_home, dopickle=True)

    print('real vs simulated')
    source = _domain_from_usenet(sraa, domain_name='real_vs_simulated', max_documents=8000,
                                 positive=['realaviation'],
                                 negative=['simaviation'])
    target = _domain_from_usenet(sraa, domain_name='real_vs_simulated_target', max_documents=8000,
                                 positive=['realauto'],
                                 negative=['simauto'])
    yield source, target, '0. real vs simulated'

    print('auto vs aviation')
    source = _domain_from_usenet(sraa, domain_name='auto_vs_aviation', max_documents=8000,
                                 positive=['simauto'],
                                 negative=['simaviation'])
    target = _domain_from_usenet(sraa, domain_name='auto_vs_aviation_target', max_documents=8000,
                                 positive=['realauto'],
                                 negative=['realaviation'])
    yield source, target, '1. auto vs aviation'


def Topic_task_generator(reuters_home, sraa_home, twenty_home):
    # Generates the tasks for cross-domain classification by topic
    for source, target, task in Reuters_task_generator(reuters_home): yield source, target, task, 'Reuters'
    for source, target, task in SRAA_task_generator(sraa_home): yield source, target, task, 'SRAA'
    for source, target, task in TwentyNews_task_generator(twenty_home): yield source, target, task, 'TwentyNews'
