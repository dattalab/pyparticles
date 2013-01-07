from __future__ import division
import numpy as np
import os

import pose_models

import particle_filter as pf
import predictive_models as pm
import predictive_distributions as pd

'''
The classes in this file handle experiment setups, including the choice of pose
model, data (sub)sequence, and particle configuration and parameters. These
classes do NOT specify the numbers of particles; those are left as an argument
to the functions in run.py since they may change more frequently than the
experiment configurations described here.

An Experiment object in this file must have the following members:

    - name (string)
    - datapath (string)
    - frame_range (tuple of two ints)

    - get_initial_particles(num_particles_firststep)
    - get_log_likelihood(ms,xytheta)
    - first_step_done(particlefilter) (for any changes after first step)
'''

class Experiment1(object):
    name = 'theta in sideinformation'
    datapath = os.path.join(os.path.dirname(__file__),"Test Data")
    frame_range = (701,750)

    # x, y, theta_yaw
    _initial_xytheta_noisechol = np.diag((1.,1.,6.))
    _subsequent_xytheta_noisechol = _initial_xytheta_noisechol
    # z, theta_roll, s_w, s_l, s_h, phi_z^1, phi_z^2, phi_y^3, phi_z^3, phi_y^4, phi_z^4,...
    _initial_randomwalk_noisechol = np.diag((3.,0.01,2.,2.,10.,) + (20.,)*(2+2*3))
    _subsequent_randomwalk_noisechol = np.diag((2.,0.01,0.2,0.2,1.0,) + (5.,)*(2+2*3))

    def __init__(self):
        self.pose_model = pose_models.PoseModel3()

        # these copies serve as references to the particles' current noises
        # having them in separate memory keeps them distinct from this class's
        # parameters (and any other instance's changes)
        self._xytheta_noisechol = self._initial_xytheta_noisechol.copy()
        self._randomwalk_noisechol = self._initial_randomwalk_noisechol.copy()

    def get_initial_particles(self,num_particles_firststep):
        return [pf.AR(
                    numlags=1,
                    previous_outputs=(self.pose_model.default_particle_pose,),
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
            return ms.get_likelihood(im,particle_data=self.pose_model.expand_poses(poses),
                x=xytheta[stepnum,0],y=xytheta[stepnum,1],theta=xytheta[stepnum,2])/500.
        return log_likelihood

    def first_step_done(self,particlefilter):
        self._xytheta_noisechol[:] = self._subsequent_xytheta_noisechol[:]
        self._randomwalk_noisechol[:] = self._subsequent_randomwalk_noisechol[:]

class Experiment2(Experiment1):
    name = 'theta in random walk'
    frame_range = (5,2000)

    _initial_xytheta_noisechol = np.diag((2.,2.))
    _subsequent_xytheta_noisechol = _initial_xytheta_noisechol

    # TODO theta proposals larger?
    _initial_randomwalk_noisechol = np.diag((7.,3.,0.01,2.,2.,10.,) + (20.,)*(2+2*3))
    _subsequent_randomwalk_noisechol = np.diag((3.,2.,0.01,0.2,0.2,1.0,) + (6.,)*(2+2*3))

    def __init__(self):
        self.pose_model = pose_models.PoseModel4()

        from renderer.load_data import load_behavior_data
        self.pose_model.default_renderer_pose = \
                self.pose_model.default_renderer_pose._replace(theta_yaw=
                        load_behavior_data(self.datapath,self.frame_range[1]+1,'angle')[self.frame_range[0]])
        self.pose_model.default_particle_pose = \
                self.pose_model.default_particle_pose._replace(theta_yaw=
                        load_behavior_data(self.datapath,self.frame_range[1]+1,'angle')[self.frame_range[0]])

        self._xytheta_noisechol = self._initial_xytheta_noisechol.copy()
        self._randomwalk_noisechol = self._initial_randomwalk_noisechol.copy()

    def get_initial_particles(self,num_particles_firststep):
        return [pf.AR(
                    numlags=1,
                    previous_outputs=(self.pose_model.default_particle_pose,),
                    baseclass=lambda: \
                        pm.Concatenation(
                            components=(
                                pm.SideInfo(noiseclass=lambda: pd.FixedNoise(self._xytheta_noisechol)),
                                pm.RandomWalk(noiseclass=lambda:pd.FixedNoise(self._randomwalk_noisechol))
                            ),
                            arggetters=(
                                lambda d: {'sideinfo':d['sideinfo'][:2]},
                                lambda d: {'lagged_outputs': map(lambda x: x[2:],d['lagged_outputs'])})
                            )
                ) for itr in range(num_particles_firststep)]

    def get_log_likelihood(self,ms,xytheta):
        def log_likelihood(stepnum,im,poses):
            return ms.get_likelihood(im,particle_data=self.pose_model.expand_poses(poses),
                x=xytheta[stepnum,0],y=xytheta[stepnum,1],theta=xytheta[stepnum,2])/1000.
        return log_likelihood

    def first_step_done(self,particlefilter):
        self._xytheta_noisechol[:] = self._subsequent_xytheta_noisechol[:]
        self._randomwalk_noisechol[:] = self._subsequent_randomwalk_noisechol[:]


class ContinuedExperiment2(Experiment2):
    frame_range = (1001,2000)

    def __init__(self):
        super(ContinuedExperiment2,self).__init__()
        self._xytheta_noisechol = self._subsequent_xytheta_noisechol.copy()
        self._randomwalk_noisechol = self._subsequent_randomwalk_noisechol.copy()

    def get_initial_particles(self,num_particles_firststep):
        initial = np.load('alltracks_end_firstrun.npy')
        return [pf.AR(
                    numlags=1,
                    previous_outputs=(initial_i,),
                    baseclass=lambda: \
                        pm.Concatenation(
                            components=(
                                pm.SideInfo(noiseclass=lambda: pd.FixedNoise(self._xytheta_noisechol)),
                                pm.RandomWalk(noiseclass=lambda:pd.FixedNoise(self._randomwalk_noisechol))
                            ),
                            arggetters=(
                                lambda d: {'sideinfo':d['sideinfo'][:2]},
                                lambda d: {'lagged_outputs': map(lambda x: x[2:],d['lagged_outputs'])})
                            )
                ) for initial_i in initial]

    def first_step_done(self,particlefilter):
        pass

class Experiment2Full(Experiment2):
    frame_range = (5,2000)

class Experiment3(object):
    name = 'all in random walk'
    datapath = os.path.join(os.path.dirname(__file__),"Test Data","Blurred Edge")
    frame_range = (5,1000)
    # frame_range = (400,450)

    _initial_randomwalk_noisechol = np.diag((3.,3.,7.,3.,0.01,2.,2.,10.,) + (20.,)*(2+2*3))
    _subsequent_randomwalk_noisechol = np.diag((1.,1.,2.,0.5,0.01,0.1,0.1,0.5,) + (1.5,)*(2+2*3))

    def __init__(self):
        self.pose_model = pose_models.PoseModel4()

        from renderer.load_data import load_behavior_data
        x_start,y_start = load_behavior_data(self.datapath,self.frame_range[0]+1,'centroid')[-1]
        theta_start = load_behavior_data(self.datapath,self.frame_range[0]+1,'angle')[-1]
        self.pose_model.default_renderer_pose = \
                self.pose_model.default_renderer_pose._replace(
                        theta_yaw=theta_start,
                        x=x_start,
                        y=y_start) # not really necessary
        self.pose_model.default_particle_pose = \
                self.pose_model.default_particle_pose._replace(
                        theta_yaw=theta_start,
                        x=x_start,
                        y=y_start)

        self._randomwalk_noisechol = self._initial_randomwalk_noisechol.copy()

    def get_initial_particles(self,num_particles_firststep):
        return [pf.AR(
                    numlags=1,
                    previous_outputs=(self.pose_model.default_particle_pose,),
                    baseclass=lambda: pm.RandomWalk(noiseclass=lambda:pd.FixedNoise(self._randomwalk_noisechol))
                ) for itr in range(num_particles_firststep)]

    def get_log_likelihood(self,ms,xytheta):
        def log_likelihood(stepnum,im,poses):
            return ms.get_likelihood(im,particle_data=self.pose_model.expand_poses(poses),
                x=xytheta[stepnum,0],y=xytheta[stepnum,1],theta=xytheta[stepnum,2])/500.
        return log_likelihood

    def first_step_done(self,particlefilter):
        self._randomwalk_noisechol[:] = self._subsequent_randomwalk_noisechol[:]

# TODO continuation
