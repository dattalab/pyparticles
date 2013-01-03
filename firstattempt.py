from __future__ import division
import numpy as np
na = np.newaxis
from warnings import warn
import os

from renderer.load_data import load_behavior_data
from renderer.renderer import MouseScene

import predictive_models as pm
import predictive_distributions as pd
import particle_filter as pf
from util.text import progprint_xrange



################
#  parameters  #
################

### experiment parameters
# TODO we should add local symbolic links along with a .gitignore

# datapath = os.path.join(os.path.dirname(__file__),"Test Data/Mouse No Median Filter, No Dilation")
datapath = os.path.join(os.path.dirname(__file__),"Test Data")
# datapath = "/Users/Alex/Dropbox/Science/Datta lab/Posture Tracking/Test Data"
frame_indices = (5,180)
use_mouse_model_version = 2 # 1, 2 or 3 for now, also affects _definitions below

# TODO just put mousemodel stuff in their own files, import whichever model!

if use_mouse_model_version == 1:
    scenefilepath = "renderer/data/mouse_mesh_low_poly.npz"
    expanded_pose_tuple_len = 3+3*9
    particle_pose_tuple_len = 3+2*9
elif use_mouse_model_version == 2:
    scenefilepath = "renderer/data/mouse_mesh_low_poly2.npz"
    expanded_pose_tuple_len = 8+3*6
    particle_pose_tuple_len = 8+2*6
elif use_mouse_model_version == 3:
    scenefilepath = "renderer/data/mouse_mesh_low_poly3.npz"
    expanded_pose_tuple_len = 8+3*5
    particle_pose_tuple_len = 7+2*5 - 2


### computation parameters
msNumRows, msNumCols = (32,32)
num_particles = msNumRows*msNumCols*5

####################
#  global objects  #
####################

# these global objects are shared for several purposes and convenient for
# interactive access. making them lazily loaded provides fast importing

ms = None # MouseScene object, global so that it's only built once
xytheta = images = None # offset sideinfo sequence and image sequence

def _build_mousescene():
    global ms
    if ms is None:
        ms = MouseScene(scenefilepath, mouse_width=80, mouse_height=80, \
                numCols=msNumCols, numRows=msNumRows, useFramebuffer=True,showTiming=False)
        ms.gl_init()

def _load_data_and_sideinfo():
    global xytheta, images
    if xytheta is None or images is None:
        xy = load_behavior_data(datapath,sum(frame_indices),'centroid')[frame_indices[0]:]
        theta = load_behavior_data(datapath,sum(frame_indices),'angle')[frame_indices[0]:]
        xytheta = np.concatenate((xy,theta[:,na]),axis=1)
        images = load_behavior_data(datapath,sum(frame_indices),'images').astype('float32')[frame_indices[0]:]
        images = np.array([image.T[::-1,:].astype('float32') for image in images])/354.0

##################################
#  log_likelihood and rendering  #
##################################

def _expand_poses1(poses):
    poses = np.array(poses)
    assert poses.ndim == 2

    expandedposes = np.zeros((poses.shape[0],expanded_pose_tuple_len))
    expandedposes[:,3::3] = ms.get_joint_rotations()[0,:,0] # x angles are fixed
    expandedposes[:,:3] = poses[:,:3] # copy in xytheta
    expandedposes[:,4::3] = poses[:,3::2] # copy in y angles
    expandedposes[:,5::3] = poses[:,4::2] # copy in z angles
    return expandedposes

def _expand_poses2(poses):
    poses = np.array(poses)
    assert poses.ndim == 2

    expandedposes = np.zeros((poses.shape[0],8+3*ms.num_bones))
    expandedposes[:,8::3] = ms.get_joint_rotations()[0,:,0] # x angles are fixed
    expandedposes[:,:2] = poses[:,:2] # x y
    expandedposes[:,2] = poses[:,3] # z
    expandedposes[:,3] = poses[:,2] # theta yaw
    expandedposes[:,4] = poses[:,4] # copy in theta roll
    expandedposes[:,5:8] = poses[:,5:8] # copy in width, length and height scales
    expandedposes[:,9::3] = poses[:,8::2] # copy in y angles
    expandedposes[:,10::3] = poses[:,9::2] # copy in z angles
    return expandedposes

def _expand_poses3(poses):
    poses = np.array(poses)
    assert poses.ndim == 2

    expandedposes = np.zeros((poses.shape[0],8+3*ms.num_bones))
    default_rotations = ms.get_joint_rotations()
    expandedposes[:,8::3] = default_rotations[0,:,0] # x angles are fixed
    expandedposes[:,9:9+6:3] = default_rotations[0,:2,1] # the first two y angles are fixed
    expandedposes[:,:2] = poses[:,:2] # x y
    expandedposes[:,2] = poses[:,3] # z
    expandedposes[:,3] = poses[:,2] # theta yaw
    expandedposes[:,5:8] = poses[:,4:7] # copy in width, length and height scales
    expandedposes[:,10:10+6:3] = poses[:,7:9] # copy in z angles (first two joints DO have z-angles)
    expandedposes[:,9+6::3] = poses[:,9::2] # copy in y angles (but not the first two joints)
    expandedposes[:,10+6::3] = poses[:,10::2] # copy in the rest of the z angles
    return expandedposes

if use_mouse_model_version == 1:
    _expand_poses = _expand_poses1
elif use_mouse_model_version == 2:
    _expand_poses = _expand_poses2
elif use_mouse_model_version == 3:
    _expand_poses = _expand_poses3

def log_likelihood(stepnum,im,poses):
    _build_mousescene(), _load_data_and_sideinfo()
    return ms.get_likelihood(im,particle_data=_expand_poses(poses),
            x=xytheta[stepnum,0],y=xytheta[stepnum,1],theta=xytheta[stepnum,2])

def render(stepnum,poses, is_expanded=True):
    warn('untested')
    _build_mousescene(), _load_data_and_sideinfo()

    def get_poses(poses):
        if is_expanded:
            return poses
        else:
            return expand_poses(poses)

    return ms.get_likelihood(np.zeros((ms.mouse_width, ms.mouse_height)), particle_data=get_poses(poses),
            x=xytheta[stepnum,0],y=xytheta[stepnum,1],theta=xytheta[stepnum,2],
            return_posed_mice=True)[1]


#############
#  running  #
#############

def run_randomwalk_fixednoise_sideinfo(cutofffactor):
    _build_mousescene(), _load_data_and_sideinfo()

    # TODO ugly ugly ugly
    initial_pose = np.zeros(particle_pose_tuple_len)
    default_rotations = ms.get_joint_rotations()
    if use_mouse_model_version == 1:
        initial_pose[3::2] = default_rotations[0,:,1] # y angles
        initial_pose[4::2] = default_rotations[0,:,2] # z angles
        xytheta_noisechol = np.diag( (1e-3,)*2 + (1e-3,) )
        joints_noisechol = np.diag( (10.,10.) + (10.,)*(2*8) )
    elif use_mouse_model_version == 2:
        initial_pose[5:8] = (18,18,200)
        initial_pose[8::2] = default_rotations[0,:,1] # y angles
        initial_pose[9::2] = default_rotations[0,:,2] # z angles
        # x,y,theta
        xytheta_noisechol = np.diag( (1e-3,)*2 + (1e-3,) )
        # z, theta_roll, width, length, height, thetay1, thetaz1, thetay2, thetaz2, ...
        joints_noisechol = np.diag( (10.,) + (1e-3,) + (2.,)*2 + (10.,) + (15.,)*(2*6) )
    elif use_mouse_model_version == 3:
        # x, y, theta, z, width, length, height, thetaz1, thetaz2, thetay3, thetaz3, thetay4, thetaz4, ...
        initial_pose[4:7] = (18,18,200)
        initial_pose[7:9] = default_rotations[0,:2,2] # first two joints have z-angles
        initial_pose[9::2] = default_rotations[0,2:,1] # y angles for every joint after the first two
        initial_pose[10::2] = default_rotations[0,2:,2] # z angles for every joint after the first two
        # x, y, theta
        xytheta_noisechol = np.diag( (1e-3,)*2 + (1e-3,) )
        # z, width, length, height, thetaz1, thetaz2, thetay3, thetaz3, thetay4, thetaz4, ...
        joints_noisechol = np.diag( (10.,) + (2.,)*2 + (10.,) + (15.,)*(2*5 - 2) )


    initial_particles = [
            pf.AR(
                    numlags=1,
                    previous_outputs=(initial_pose,),
                    baseclass=lambda: \
                        pm.Concatenation(
                            components=(
                                pm.SideInfo(noiseclass=lambda: pd.FixedNoise(xytheta_noisechol)),
                                pm.RandomWalk(noiseclass=lambda:pd.FixedNoise(joints_noisechol))
                            ),
                            arggetters=(
                                lambda d: {'sideinfo':d['sideinfo']},
                                lambda d: {'lagged_outputs': map(lambda x: x[3:],d['lagged_outputs'])})
                            )
                ) for itr in range(num_particles*5)] # NOTE: 5x initial particles

    # create particle filter
    particlefilter = pf.ParticleFilter(particle_pose_tuple_len,num_particles/cutofffactor,log_likelihood,initial_particles)

    # first step is special
    particlefilter.step(images[0],sideinfo=xytheta[0])

    # change the number of particles and the noises
    if use_mouse_model_version == 1:
        xytheta_noisechol[:,:] = np.diag( (1e-2,)*2 + (1e-2,) )
        joints_noisechol[:,:] = np.diag( (3.,3.) + (3.,) * (2*8) )
    elif use_mouse_model_version == 2:
        # x,y,theta
        xytheta_noisechol = np.diag( (1e-2,)*2 + (1e-2,) )
        # z, theta_roll, width, length, height, thetay1, thetaz1, thetay2, thetaz2, ...
        joints_noisechol = np.diag( (1.,) + (1e-3,) + (1e-2,)*3 + (3.,)*(2*6) )
    elif use_mouse_model_version == 3:
        xytheta_noisechol[:,:] = np.diag( (1e-2,)*2 + (1e-2,) )
        joints_noisechol = np.diag( (1.,) + (1e-2,)*3 + (3.,)*(2*5 - 2) )

    particlefilter.change_numparticles(num_particles)

    for i in progprint_xrange(1,5):
        particlefilter.step(images[i],sideinfo=xytheta[i])

    return particlefilter

###########
#  utils  #
###########

def topk(items,scores,k):
    return [items[idx] for idx in np.argsort(scores)[:-(k+1):-1]]

def meantrack(particles,weights):
    track = np.array(particles[0].track)*weights[0,na]
    for p,w in zip(particles[1:],weights[1:]):
        track += np.array(p.track) * w
    return track

##########
#  main  #
##########

if __name__ == '__main__':
    result = run_randomwalk_fixednoise_sideinfo(1)
    np.save('top5tracks',np.array([_expand_poses(p.track) for p in topk(result.particles,result.weights_norm,5)]))
    np.save('meantrack',_expand_poses(meantrack(result.particles,result.weights_norm)))

