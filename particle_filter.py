from __future__ import division
import numpy as np

class ParticleFilter(object):
    def __init__(self,dim,n_particles,cutoff,log_likelihood_fn,particle_factory):
        self.particles = [particle_factory() for idx in range(n_particles)]
        self.locs = np.empty((n_particles,dim))
        self.log_weights = np.zeros(n_particles)
        self.weights_norm = np.ones(n_particles)
        self.log_likelihood_fn = log_likelihood_fn
        self.cutoff = cutoff

    def step(self,data,*args,**kwargs):
        for idx, particle in enumerate(self.particles):
            self.locs[idx] = particle.sample_next(*args,**kwargs)
        self.log_weights += self.log_likelihood_fn(data,self.locs)

        if self._Neff() < self.cutoff:
            print 'resampling'
            self._resample()
            self._Neff()

        print ''

    def _Neff(self):
        self.weights_norm = np.exp(self.log_weights - np.logaddexp.reduce(self.log_weights))
        self.weights_norm /= self.weights_norm.sum()
        Neff = 1./np.sum(self.weights_norm**2)
        print Neff
        return Neff

    def _resample(self):
        sources = np.random.multinomial(1,self.weights_norm,size=len(self.particles)).argmax(1)
        self.particles = [self.particles[idx].copy() for idx in sources]
        self.log_weights = (np.logaddexp.reduce(self.log_weights) - np.log(len(self.particles))) \
                * np.ones(len(self.particles))

