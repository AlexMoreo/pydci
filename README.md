# Distributional Correspondence Indexing (DCI)
## (A Python Implementation)

This python implementation of the Distributional Correspondence Indexig (DCI) for domain adaptation allows to replicate experiments for:

* **Cross-domain adaptation**: using the *MultiDomainSentiment* (MDS) dataset
  
* **Cross-lingual adaptation**: using the *Webis-CLS-10* dataset

## Requirements

This package has been tested with the following environment (though it might work with older versions too).
* Python 3.5.2
* Numpy 1.15.2
* Scipy 1.0.0
* Sklearn 0.19.1

## Replicate the experiments:

There is one script devoted to reproduce each of the experiments reported in <link>. To replicate, e.g., the cross-domain adaptation experiments, simply run:

```
cd src
python cross_domain_sentiment.py
```


