from __future__ import division
import numpy as np
na = np.newaxis
from collections import deque
import abc
from warnings import warn

from util.general import ibincount

DEBUG = True

class ParticleFilter(object):
    def __init__(self,ndim,cutoff,log_likelihood_fn,initial_particles):
        assert len(initial_particles) > 0

        self.particles = initial_particles
        self.log_likelihood_fn = log_likelihood_fn
        self.cutoff = cutoff

        self.locs = np.empty((len(initial_particles),ndim))
        self.log_weights = np.zeros(len(initial_particles))
        self.weights_norm = np.ones(len(initial_particles))

        self.Nsurvive_history = []
        self.Neff_history = []
        self.numsteps = 0

    def step(self,data,resample_method='independent',particle_kwargs={}):
        for idx, particle in enumerate(self.particles):
            self.locs[idx] = particle.sample_next(**particle_kwargs)
        self.log_weights += self.log_likelihood_fn(self.numsteps,data,self.locs)

        if self._Neff < self.cutoff:
            self._resample(resample_method)
            resampled = True
        else:
            resampled = False

        self.numsteps += 1

        return resampled

    def change_numparticles(self,newnum,resample_method='independent'):
        if newnum != len(self.particles):
            self._resample(num=newnum)

    def inject_particles(self,particles_to_inject,particle_kwargs={}):
        self.locs = np.concatenate((self.locs,[p.sample_next(**particle_kwargs) for p in particles_to_inject]))
        self.particles += particles_to_inject

    @property
    def _Neff(self):
        self.weights_norm = np.exp(self.log_weights - np.logaddexp.reduce(self.log_weights))
        self.weights_norm /= self.weights_norm.sum()
        Neff = 1./np.sum(self.weights_norm**2)

        self.Neff_history.append((self.numsteps,Neff))

        if DEBUG:
            print Neff

        return Neff

    def _resample(self,method,num=None):
        num = (num if num is not None else len(self.particles))

        assert method in ['lowvariance','independent']
        if method is 'lowvariance':
            sources = self._lowvariance_sources(num)
        if method is 'independent':
            sources = self._independent_sources(num)

        self.log_weights = np.repeat(np.logaddexp.reduce(self.log_weights) - np.log(num),num)
        self.weights_norm = np.repeat(1./num, num)

        self.Nsurvive_history.append((self.numsteps,len(np.unique(sources))))

        if DEBUG:
            print self.Nsurvive_history[-1][1]

    def _independent_sources(self,num):
        return ibincount(np.random.multinomial(num,self.weights_norm))

    def _lowvariance_sources(self,num):
        r = np.random.rand()/num
        bins = np.concatenate(((0,),np.cumsum(self.weights_norm)))
        return ibincount(np.histogram(r*(np.arange(1,1+num)),bins)[0])

    def __getstate__(self):
        result = self.__dict__.copy()
        del result['log_likelihood_fn']
        return result


######################
#  Particle objects  #
######################

class Particle(object):
    __metaclass__= abc.ABCMeta
    # NOTE: also needs a 'track' instance member

    @abc.abstractmethod
    def sample_next(self,*args,**kwargs):
        pass

    @abc.abstractmethod
    def copy(self):
        pass


class BasicParticle(Particle):
    def __init__(self,baseclass):
        self.sampler = baseclass()
        self.track = []

    def sample_next(self,*args,**kwargs):
        self.track.append(self.sampler.sample_next(*args,**kwargs))
        return self.track[-1]

    def copy(self):
        new = self.__new__(self.__class__)
        new.track = self.track[:] # shallow copy
        new.sampler = self.sampler.copy()
        return new

    def __str__(self):
        return '%s(%s)' % (self.__class__.__name__,self.sampler.__str__())

    def __getstate__(self):
        return {'track':self.track}


class AR(BasicParticle):
    def __init__(self,numlags,baseclass,previous_outputs=[],initial_baseclass=None):
        assert len(previous_outputs) == numlags or initial_baseclass is not None
        super(AR,self).__init__(baseclass)
        self.lagged_outputs = deque(previous_outputs,maxlen=numlags)
        if len(self.lagged_outputs) < numlags:
            self.initial_sampler = initial_baseclass()

    def sample_next(self,*args,**kwargs):
        if len(self.lagged_outputs) < self.lagged_outputs.maxlen:
            out = self.initial_sampler.sample_next(lagged_outputs=self.lagged_outputs,*args,**kwargs)
        else:
            out = self.sampler.sample_next(lagged_outputs=self.lagged_outputs,*args,**kwargs)
        self.lagged_outputs.appendleft(out)
        self.track.append(out)
        return out

    def copy(self):
        new = super(AR,self).copy()
        new.lagged_outputs = self.lagged_outputs.__copy__()
        if len(self.lagged_outputs) < self.lagged_outputs.maxlen:
            new.initial_sampler = self.initial_sampler.copy()
        return new

###############
#  Utilities  #
###############

def topktracks(pf,k):
    indices = np.argsort(pf.weights_norm)[:-(k+1):-1]
    return np.array([pf.particles[i].track for i in indices]), pf.weights_norm[indices]

def meantrack(pf):
    track = np.array(pf.particles[0].track)*pf.weights_norm[0,na]
    for p,w in zip(pf.particles[1:],pf.weights_norm[1:]):
        track += np.array(p.track) * w
    return track
