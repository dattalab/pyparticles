from __future__ import division
import numpy as np

import mouse_models

import particle_filter as pf
import predictive_models as pm
import predictive_distributions as pd

'''
The classes in this file handle experiment setups, including the choice of mouse
model, data (sub)sequence, and particle parameters. These classes do NOT specify
the numbers of particles; those are left as an argument to the functions in
run.py since they may change more frequently than the experiment configurations
described here.

An Experiment object in this file must have the following members:

    - datapath (string)
    - frame_range (tuple of two ints)

    - get_initial_particles(num_particles_firststep)
    - get_log_likelihood(ms,xytheta)
    - first_step_done(particlefilter) (for any changes after first step)
'''

class Experiment1(object):
    datapath = "Test Data"
    frame_range = (5,180)

    _initial_xytheta_noisechol = None
    _initial_joints_noisechol = None

    _subsequent_xytheta_noisechol = None
    _subsequent_joints_noisechol = None

    def __init__(self):
        self.mouse_model = mouse_models.Model3()

        # these copies serve as references to the particles' current noises
        # having them in separate memory keeps them distinct from this class's
        # parameters (and any other instance's changes)
        self._xytheta_noisechol = self._initial_xytheta_noisechol.copy()
        self._joints_noisechol = self._initial_joints_noisechol.copy()

    def get_initial_particles(self,num_particles_firststep):
        return [pf.AR(
                    numlags=1,
                    previous_outputs=(self.mouse_model.default_particle_pose,),
                    baseclass=lambda: \
                        pm.Concatenation(
                            components=(
                                pm.SideInfo(noiseclass=lambda: pd.FixedNoise(self._xytheta_noisechol)),
                                pm.RandomWalk(noiseclass=lambda:pd.FixedNoise(self._joints_noisechol))
                            ),
                            arggetters=(
                                lambda d: {'sideinfo':d['sideinfo']},
                                lambda d: {'lagged_outputs': map(lambda x: x[3:],d['lagged_outputs'])})
                            )
                ) for itr in range(num_particles_firststep)]

    def get_log_likelihood(self,ms,xytheta):
        def log_likelihood(stepnum,im,poses):
            return ms.get_likelihood(im,particle_data=self.mouse_model.expand_poses(poses),
                x=xytheta[stepnum,0],y=xytheta[stepnum,1],theta=xytheta[stepnum,2])
        return log_likelihood

    def first_step_done(self,particlefilter):
        self._xytheta_noisechol[:] = self._subsequent_xytheta_noisechol[:]
        self._joints_noisechol[:] = self._subsequent_joints_noisechol[:]

