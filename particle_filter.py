from __future__ import division
import numpy as np
from collections import deque
import abc

class ParticleFilter(object):
    def __init__(self,ndim,cutoff,log_likelihood_fn,initial_particles):
        assert len(initial_particles) > 0
        n_particles = len(initial_particles)
        self.particles = initial_particles
        self.locs = np.empty((n_particles,ndim))
        self.log_weights = np.zeros(n_particles)
        self.weights_norm = np.ones(n_particles)
        self.log_likelihood_fn = log_likelihood_fn
        self.cutoff = cutoff
        self.Neff_history = []

    def step(self,data,*args,**kwargs):
        for idx, particle in enumerate(self.particles):
            self.locs[idx] = particle.sample_next(*args,**kwargs)
        self.log_weights += self.log_likelihood_fn(data,self.locs)

        if self._Neff() < self.cutoff:
            self._resample()
            self._Neff()

    def _Neff(self):
        self.weights_norm = np.exp(self.log_weights - np.logaddexp.reduce(self.log_weights))
        self.weights_norm /= self.weights_norm.sum()
        Neff = 1./np.sum(self.weights_norm**2)
        self.Neff_history.append(Neff)
        return Neff

    def _resample(self):
        sources = np.random.multinomial(1,self.weights_norm,size=len(self.particles)).argmax(1)
        self.particles = [self.particles[idx].copy() for idx in sources]
        self.log_weights = (np.logaddexp.reduce(self.log_weights) - np.log(len(self.particles))) \
                * np.ones(len(self.particles))

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

    def __getattr__(self,name):
        # pull up sampler's members for convenience
        return getattr(self.sampler,name)


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

