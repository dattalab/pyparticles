from __future__ import division
import numpy as np
na = np.newaxis

from renderer.load_data import load_behavior_data
from renderer import MouseScene

import predictive_models as pm
import predictive_distributions as pd
import particle_filter as pf

datapath = "/Users/mattjj/Dropbox/Test Data/"
scenefilepath = "renderer/data/mouse_mesh_low_poly.npz"

def run_momentum_fixednoise(cutoff):
    # load data
    data = load_behavior_data(datapath,120,"images")
    data = np.array([np.rot90(i) for i in data])
    data /= 354.0;

    # make mousescene object, pull out likelihood function
    num_particles = 32**2
    numCols = int(np.sqrt(num_particles))
    numRows = numCols
    ms = MouseScene(scenefilepath, mouse_width=80, mouse_height=80, \
            scale = 2.0, \
            numCols=numCols, numRows=numRows, useFramebuffer=True)
    ms.gl_init()

    likelihood = ms.get_likelihood

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
    particlefilter = pf.ParticleFilter(2,num_particles,cutoff,likelihood,particle_factory)

    # loop!!!
    for f in range(120):
        particlefilter.step(data[i])


def run_momentum_fixednoise_sideinfo(nparticles,cutoff):
    invwishparams = (20,20*np.diag( (20.,20.,5.,) + (45.,)*16 )**2) # TODO mixture
    # TODO get tracked x,y,theta for data-dependent version
    pass

