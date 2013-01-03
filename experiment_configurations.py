from __future__ import division
import numpy as np

import mouse_models

import particle_filter as pf
import predictive_models as pm
import predictive_distributions as pd

'''
The classes in this file handle experiment setups, including the choice of mouse
model, data (sub)sequence, and particle configuration and parameters. These
classes do NOT specify the numbers of particles; those are left as an argument
to the functions in run.py since they may change more frequently than the
experiment configurations described here.

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

    # chosen from lines 774 and 781 of
    # https://github.com/mattjj/hsmm-particlefilters/blob/ab175f229e219f5117fde5ce76921e0014419180/renderer/renderer.py
    # x, y, theta_yaw
    _initial_xytheta_noisechol = np.diag((1.,1.,3.))

    # chosen from lines 777, 788-790, and 801-802 of
    # https://github.com/mattjj/hsmm-particlefilters/blob/ab175f229e219f5117fde5ce76921e0014419180/renderer/renderer.py
    # z, s_w, s_l, s_h, phi_z^1, phi_z^2, phi_y^3, phi_z^3, phi_y^4, phi_z^4,...
    _initial_randomwalk_noisechol = np.diag((3.,2.,2.,10.,) + (20.)*(2+2*3))

    _subsequent_xytheta_noisechol = _initial_xytheta_noisechol
    _subsequent_randomwalk_noisechol = _initial_randomwalk_noisechol / 2.5 # TODO guessed

    def __init__(self):
        self.mouse_model = mouse_models.Model3()

        # these copies serve as references to the particles' current noises
        # having them in separate memory keeps them distinct from this class's
        # parameters (and any other instance's changes)
        self._xytheta_noisechol = self._initial_xytheta_noisechol.copy()
        self._randomwalk_noisechol = self._initial_randomwalk_noisechol.copy()

    def get_initial_particles(self,num_particles_firststep):
        return [pf.AR(
                    numlags=1,
                    previous_outputs=(self.mouse_model.default_particle_pose,),
                    baseclass=lambda: \
                        pm.Concatenation(
                            components=(
                                pm.SideInfo(noiseclass=lambda: pd.FixedNoise(self._xytheta_noisechol)),
                                pm.RandomWalk(noiseclass=lambda:pd.FixedNoise(self._randomwalk_noisechol))
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
        self._randomwalk_noisechol[:] = self._subsequent_randomwalk_noisechol[:]

