# Distributional Correspondence Indexing (DCI)
## (A Python Implementation)

This python implementation of the Distributional Correspondence Indexig (DCI) for domain adaptation allows to replicate experiments for:

* **Cross-domain adaptation (by Sentiment)**: using the *MultiDomainSentiment* (MDS) dataset
  
* **Cross-lingual adaptation (by Sentiment)**: using the *Webis-CLS-10* dataset

* **Cross-domain adaptation (by Topic)**: using the *Reuters, SRAA,* and *20Newsgroups* datasets

## Publications

The main method is described in:

* Distributional Correspondence Indexing for Cross-Lingual and Cross-Domain Sentiment Classification. A Moreo, A Esuli, F Sebastiani, Journal of Artificial Intelligence Research 55, 131-163 [pdf](https://www.jair.org/index.php/jair/article/view/10977)
* Distributional Correspondence Indexing for Cross-Language Text Categorization, A Esuli, A Moreo, Advances in Information Retrieval, 104-109 [pdf](http://www.esuli.it/publications/ECIR2015.pdf)

This implementation (pyDCI) is described and tested in :

* Revisiting distributional correspondence indexing: A Python reimplementation and new experiments, A Moreo, A Esuli, F Sebastiani, arXiv preprint arXiv:1810.09311 [pdf](https://arxiv.org/abs/1810.09311)

Publications based on DCI:

* Lost in Transduction: Transductive Transfer Learning in Text Classification, A Moreo, A Esuli, F Sebastiani, ACM Transactions on Knowledge Discovery from Data. Forthcoming.
* Cross-lingual sentiment quantification, A Esuli, A Moreo, F Sebastiani, IEEE Intelligent Systems 35 (3), 106-114 [pdf](https://ieeexplore.ieee.org/abstract/document/9131128/)
* Transductive Distributional Correspondence Indexing for Cross-Domain Topic Classification, A Moreo, A Esuli, and F Sebastiani IIR. 2016. [pdf](http://ceur-ws.org/Vol-1653/paper_5.pdf)

Other related publications:

* Heterogeneous Document Embeddings for Cross-Lingual Text Classification, A Moreo, A Pedrotti, F Sebastiani, SAC 2021, 36th ACM Symposium On Applied Computing, Gwangju, KR, 685-688
* Funnelling: A New Ensemble Method for Heterogeneous Transfer Learning and Its Application to Cross-Lingual Text Classification, A Esuli, A Moreo, F Sebastiani, ACM Transactions on Information Systems (TOIS) 37 (3), 1-30 [pdf](https://dl.acm.org/doi/abs/10.1145/3326065)


## Requirements

This package has been tested with the following environment (though it might work with older versions too).
* Python 3.5.2
* Numpy 1.15.2
* Scipy 1.0.0
* Sklearn 0.19.1
* Pandas 0.20.3
* SVMlight (for transductive inference)

## Replicate the experiments:

First, clone the repo by typing:

```
git clone https://github.com/AlexMoreo/pydci.git
```

There is one script devoted to reproduce each of the experiments reported in https://arxiv.org/abs/1810.09311.
The scripts are very simple and they do not parse command line arguments. To replicate other configurations, just change some variables in the script (e.g., dcf= 'linear', or npivots = 900 to run PyDCI(linear) with 900 pivots) or create your own script.
To replicate, e.g., the cross-domain adaptation experiments, simply run:

```
cd pydci/src
python cross_domain_sentiment.py
```

The script will download the dataset the first time it is invoked. The script produces a result CSV file containing the classification accuracy for each (source,target) domain combination (in the case of cross-domain, also for each fold), and some timings recorded during the execution (time took to extract pivots, to project the feature spaces, to fit the classifier, and to annotate test documents). A summary of the classification accuracy is displayed when it finishes. The order of appearance of the tasks is the common order followed by most papers, that is:

```
method                       DCI(cosine)
dataset task
MDS     books dvd                 0.8225
        books electronics         0.8370
        books kitchen             0.8430
        dvd books                 0.8345
        dvd electronics           0.8545
        dvd kitchen               0.8560
        electronics books         0.8005
        electronics dvd           0.8010
        electronics kitchen       0.8780
        kitchen books             0.8075
        kitchen dvd               0.8060
        kitchen electronics       0.8600
        
Grand Totals
method   DCI(cosine)
dataset   
MDS         0.833375
```

## Transductive Adaptation:

A bunch of scripts have been added to replicate experiments using TDCI (the transductive 
variant of DCI for cross-lingual and cross-domain adaptation). Those scripts are marked
with a "_transductive" postfix. SVMlight is required in order to make them work.
The paper discussing this variant and the results is currently under review.
