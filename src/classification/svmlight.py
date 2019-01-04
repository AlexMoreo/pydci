from os.path import join, exists
import subprocess
from subprocess import PIPE, STDOUT
import tempfile
from scipy.sparse import issparse, vstack
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import dump_svmlight_file
from random import randint
import numpy as np

class SVMlight(BaseEstimator, ClassifierMixin):
    """
    A wrapper implementation for running the Joachim's SVMlight package, see: http://svmlight.joachims.org/
    """
    valid_kernels = {'linear':0, 'rbf':2}

    def __init__(self, svmlightbase='./svm_light', C=None, kernel='linear', transduction=None, verbose=True):
        """
        Instantiates a sklearn-like classifier that wraps the invokation of SVMlight
        :param svmlightbase: the path to the directory containing the executables svm_learn and svm_classify
        :param C: the parameter controlling the trade-off between training error and marging
        :param kernel: linear (default) or rbf
        :param transduction: (optional) a co-occurrence matrix of the test set for transduction. If given, the attribute
            self.is_transductive will be set to True, and the labels predictions for the test set will be made available
            by self.transduced_labels after calling the fit method (no need to call the predict)
        :param verbose: whether or not to print the svm-light messages
        """
        assert kernel in self.valid_kernels, 'unsupported kernel {}, valid ones are {}'.format(kernel, list(self.valid_kernels.keys()))
        self.tmpdir = None
        self.svmlightbase = svmlightbase
        self.svmlight_learn = join(svmlightbase, 'svm_learn')
        self.svmlight_classify = join(svmlightbase, 'svm_classify')
        self.verbose=verbose
        self.C = C
        self.kernel = kernel
        self.transduction = transduction
        self.is_transductive = (transduction is not None)
        self.__name__ = 'SVMlight-'+kernel


    def fit(self, X, y):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.model = join(self.tmpdir.name, 'model')
        traindata_path = join(self.tmpdir.name, 'train.dat')
        transductions_path = join(self.tmpdir.name, 'transd_labels.dat')

        y = y * 2 - 1 # re-code from neg=0 pos=1 --> neg=-1 pos=+1 (0 is left for unlabeled documents for transduction)
        if self.is_transductive:
            if issparse(X) and issparse(self.transduction):
                X = vstack((X, self.transduction))
            else:
                X = np.vstack((X, self.transduction))
            y = np.concatenate((y, np.zeros(self.transduction.shape[0])))

        dump_svmlight_file(X, y, traindata_path, zero_based=False)

        options = []
        if self.C is not None: options += ['-c ' + str(self.C)]
        if self.kernel is not None: options += ['-t ' + str(self.valid_kernels[self.kernel])]
        if self.is_transductive: options += ['-l ' + transductions_path]
        cmd = ' '.join([self.svmlight_learn] + options + [traindata_path, self.model])
        self.print('[Running]',cmd)
        p = subprocess.run(cmd.split(), stdout=PIPE, stderr=STDOUT)

        self.print(p.stdout.decode('utf-8'))

        if self.transduction is not None:
            self.transduced_labels = np.loadtxt(transductions_path, usecols=0,
                                                converters={0:lambda x:int(x.decode('utf-8').split(':')[1])},
                                                encoding='bytes')
            self.transduced_labels = (self.transduced_labels > 0) * 1

        return self


    def predict(self, X, y=None):
        assert self.tmpdir is not None, 'predict called before fit, or model directory corrupted'
        assert exists(self.model), 'model not found'
        if y is None:
            y = np.zeros(X.shape[0])

        random_code = '-'.join(str(randint(0,1000000)) for _ in range(5)) #this would allow to run parallel instances of predict hopefully
        predictions_path = join(self.tmpdir.name, 'predictions'+random_code+'.dat')
        testdata_path = join(self.tmpdir.name, 'test'+random_code+'.dat')
        dump_svmlight_file(X, y, testdata_path, zero_based=False)

        cmd = ' '.join([self.svmlight_classify, testdata_path, self.model, predictions_path])
        self.print('[Running]', cmd)
        p = subprocess.run(cmd.split(), stdout=PIPE, stderr=STDOUT)

        self.print(p.stdout.decode('utf-8'))

        predictions = (np.loadtxt(predictions_path) > 0)*1
        return predictions


    def score(self, X, y):
        y_ = self.predict(X)
        score_ = (y_==y).mean()
        return score_


    def print(self, msg, end='\n'):
        if self.verbose:
            print(msg,end=end)

