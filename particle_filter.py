from __future__ import division
import numpy as np

class ParticleFilter(object):
    def __init__(self,dim,n_particles,particle_factory):
        self.out = np.empty((n_particles,dim))
        self.particles = [particle_factory(self.out[itr]) for itr in range(n_particles)]
        self.log_weights = np.zeros(n_particles)

    def step(self,data):
        pass

    def _resample(self):
        pass

    def _Neff(self):
        pass

