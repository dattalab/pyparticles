from __future__ import division
import numpy as np
na = np.newaxis
from util.stats import sample_mniw

##################
#  PLACEHOLDERS  #
##################

class iidgaussian(object):
    def __init__(self):
        self.val = np.random.randn()

    def sample_next(self):
        return self.val

class iidpoisson(object):
    def __init__(self):
        self.val = np.random.poisson(5) + 1

    def sample_next(self):
        return self.val

################
#  Real Stuff  #
################

import abc

class PredictiveSampler(object):
    __metaclass__ = abc.ABCMeta

    def sample_next(self,*args,**kwargs):
        val = self._sample(*args,**kwargs)
        self._update_hypparams(val,*args,**kwargs)
        return val

    @abc.abstractmethod
    def _update_hypparams(self,x):
        pass

    @abc.abstractmethod
    def _sample(self):
        pass

class Poisson(object):
    def __init__(self,alpha_0,beta_0):
        self.alpha_n = alpha_0
        self.beta_n = beta_0

    def _update_hypparams(self,k):
        self.alpha_n += k
        self.beta_n += 1

    def _sample(self):
        return np.random.poisson(np.random.gamma(self.alpha_n,1./self.beta_n))

class MNIWAR(object):
    def __init__(self,n_0,sigma_0,M,K):
        # hyperparameters
        self.n = n_0
        self.sigma_n = self.sigma_0 = sigma_0
        self.M_n = M
        self.K_n = K

        # statistics
        self.Sytyt = K
        self.Syyt = M.dot(K)
        self.Syy = M.dot(K).dot(M.T)

    def _update_hypparams(self,y,lagged_observations):
        ylags = self._pad_ylags(lagged_observations)

        self.Syy += y * y[:,na]
        self.Sytyt += ylags * ylags[:,na]
        self.Syyt += y * ylags[:,na]

        M_n = np.linalg.solve(self.Sytyt.T,self.Syyt.T).T
        Sy_yt = self.Syy - M_n.dot(self.Sytyt.T)

        self.n += 1
        self.sigma_n = Sy_yt + self.sigma_0
        self.M_n = M_n
        self.K_n = self.Sytyt

    def _sample(self,lagged_observations):
        ylags = self._pad_ylags(lagged_observations)
        A = sample_mniw(self.n,self.sigma_n,self.M_n,self.K_n)
        return A.dot(ylags)

    def _pad_ylags(self,lagged_observations):
        ylags = np.zeros(self.K.shape[0]+1)
        temp = np.concatenate(lagged_observations)
        ylags[:temp.shape[0]] = temp
        ylags[-1] = 1
        return ylags

