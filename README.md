# Distributional Correspondence Indexing (DCI)
## (A Python Implementation)

This python implementation of the Distributional Correspondence Indexig (DCI) for domain adaptation allows to replicate experiments for:

* **Cross-domain adaptation (by Sentiment)**: using the *MultiDomainSentiment* (MDS) dataset
  
* **Cross-lingual adaptation (by Sentiment)**: using the *Webis-CLS-10* dataset

* **Cross-domain adaptation (by Topic)**: using the *Reuters, SRAA,* and *20Newsgroups* datasets

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