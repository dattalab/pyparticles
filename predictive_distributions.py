from __future__ import division
import numpy as np
na = np.newaxis
import abc, copy

from util.stats import sample_mniw

'''predictive samplers for basic distributions'''

class PredictiveDistribution(object):
    __metaclass__ = abc.ABCMeta

    def sample_next(self,*args,**kwargs):
        val = self._sample(*args,**kwargs)
        self._update_hypparams(val)
        return val

    def copy(self):
        return copy.deepcopy(self)

    @abc.abstractmethod
    def _update_hypparams(self,x):
        pass

    @abc.abstractmethod
    def _sample(self):
        pass

###############
#  Durations  #
###############

class Poisson(PredictiveDistribution):
    def __init__(self,alpha_0,beta_0):
        self.alpha_n = alpha_0
        self.beta_n = beta_0

    def _update_hypparams(self,k):
        self.alpha_n += k
        self.beta_n += 1

    def _sample(self):
        return np.random.poisson(np.random.gamma(self.alpha_n,1./self.beta_n))+1

class NegativeBinomial(PredictiveDistribution): # TODO
    pass

##################
#  Observations  #
##################

class MNIWAR(PredictiveDistribution):
    '''Conjugate Matrix-Normal-Inverse-Wishart prior'''
    def __init__(self,n_0,kappa_0,sigma_0,M,K):
        # hyperparameters
        self.n = n_0
        self.kappa_n = kappa_0
        self.sigma_0 = sigma_0
        self.M_n = M
        self.K_n = K
        self.sigma_n = sigma_0.copy()

        # statistics
        self.Sytyt = K.copy()
        self.Syyt = M.dot(K)
        self.Syy = M.dot(K).dot(M.T)

    def _update_hypparams(self,y):
        ylags = self._ylags # gets info passed from previous _sample call, state!

        self.Syy += y[:,na] * y
        self.Sytyt += ylags[:,na] * ylags
        self.Syyt += y[:,na] * ylags

        M_n = np.linalg.solve(self.Sytyt,self.Syyt.T).T
        Sy_yt = self.Syy - M_n.dot(self.Syyt.T)

        self.n += 1
        self.kappa_n += 1
        self.sigma_n = Sy_yt + self.sigma_0
        self.M_n = M_n
        self.K_n = self.Sytyt

        assert np.allclose(self.sigma_n,self.sigma_n.T) and (np.linalg.eigvals(self.sigma_n) > 0).all()
        assert np.allclose(self.K_n,self.K_n.T) and (np.linalg.eigvals(self.K_n) > 0).all()

    def _sample(self,lagged_outputs):
        ylags = self._ylags = self._pad_ylags(lagged_outputs)
        A,sigma = sample_mniw(self.n,self.kappa_n,self.sigma_n,self.M_n,np.linalg.inv(self.K_n))
        # print A
        # print sigma
        # print ''
        return A.dot(ylags) + np.linalg.cholesky(sigma).dot(np.random.randn(sigma.shape[0]))

    def _pad_ylags(self,lagged_outputs):
        ylags = np.zeros(self.M_n.shape[1])

        # plug in lagged data
        temp = np.array(lagged_outputs)
        temp.shape = (-1,)
        ylags[:temp.shape[0]] = temp

        # plug in affine drift
        ylags[-1] = 1

        return ylags


class NIWNonConjAR(PredictiveDistribution): # TODO
    # Gibbs steps on copy
    # normal, normal, inverse wishart
    # should use an IW class and a Gaussian class that do blocked gibbs
    # like pyhsmm distribution classes but which take stats, not data, and only
    # need to draw samples
    pass


class _InverseWishart(object):
    pass


class _Gaussian(object):
    pass

