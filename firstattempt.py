from __future__ import division
import numpy as np
na = np.newaxis

import predictive_models as pm
import predictive_distributions as pd
import particle_filter as pf

def run_momentum_fixednoise(nparticles,cutoff):
    # make mouse opengl object, pull out likelihood function
    likelihood = None

    # load data
    data = None

    # get initial position
    initial_pos = None

    # set up particle business
    propmatrix = np.hstack((2*np.eye(19),-1*np.eye(19)))
    noisechol = np.diag( (5.,5.,45.,) + (45.,)*16 )**2
    particle_factory = lambda: \
            pm.AR(
                    numlags=2,
                    initial_obs=(initial_pos,)*2,
                    baseclass=lambda: \
                            pm.momentum(
                                propmatrix=propmatrix,
                                noiseclass=lambda:pd.FixedNoise(noisechol)
                            )
                    )

    # create particle filter
    pf.ParticleFilter(2,nparticles,cutoff,likelihood,particle_factory)

    # loop!!!
    # TODO


def run_momentum_fixednoise_sideinfo(nparticles,cutoff):
    invwishparams = (20,20*np.diag( (20.,20.,5.,) + (45.,)*16 )**2) # TODO mixture
    # TODO get tracked x,y,theta for data-dependent version
    pass

