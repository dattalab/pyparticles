from __future__ import division
import numpy as np
na = np.newaxis
from matplotlib import pyplot as plt

from renderer.load_data import load_behavior_data
from renderer.renderer import MouseScene

import predictive_models as pm
import predictive_distributions as pd
import particle_filter as pf
from util.text import progprint_xrange

# TODO factor out 3+2*9, think about a concise spec for particlefilter setup

################
#  parameters  #
################

# experiment parameters
datapath = "/Users/mattjj/Dropbox/Test Data/"
frame_indices = (5,180)
scenefilepath = "renderer/data/mouse_mesh_low_poly.npz"

# computation parameters
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
                # or 2.0
                scale = 1.75, \
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

def _expand_poses(poses):
    expandedposes = np.zeros((poses.shape[0],3+3*9))
    expandedposes[:,3::3] = ms.get_joint_rotations()[0,:,0] # x angles are fixed
    expandedposes[:,:3] = poses[:,:3] # copy in xytheta
    expandedposes[:,4::3] = poses[:,3::2] # copy in y angles
    expandedposes[:,5::3] = poses[:,4::2] # copy in z angles

def log_likelihood(stepnum,im,poses):
    _build_mousescene(), _load_data_and_sideinfo()
    return ms.get_likelihood(im,particle_data=_expand_poses(poses),
            x=xytheta[stepnum,0],y=xytheta[stepnum,1],theta=xytheta[stepnum,2])

def render(stepnum,poses):
    _build_mousescene(), _load_data_and_sideinfo()
    return ms.get_likelihood(np.zeros((msNumRows,msNumCols)),particle_data=_expand_poses(poses),
            x=xytheta[stepnum,0],y=xytheta[stepnum,1],theta=xytheta[stepnum,2],
            return_posed_mice=True)[1]

def run_randomwalk_fixednoise_sideinfo(cutofffactor):
    _build_mousescene()
    initial_pose = np.zeros(3+2*9)
    initial_pose[3::2] = ms.get_joint_rotations()[0,:,1] # y angles
    initial_pose[4::2] = ms.get_joint_rotations()[0,:,2] # z angles

    xytheta_noisechol = np.diag( (1e-3,)*2 + (1e-3,) )
    joints_noisechol = np.diag( (10.,10.) + (10.,)*(2*8) )

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
            ) for itr in range(num_particles)]

    # create particle filter
    particlefilter = pf.ParticleFilter(3+2*9,num_particles/cutofffactor,log_likelihood,initial_particles)

    # loop!!!
    # TODO make this first step prettier, use change_num_particles
    particlefilter.step(images[0],sideinfo=xytheta[0])
    xytheta_noisechol[:,:] = np.diag( (1e-2,)*2 + (1e-2,) )
    joints_noisechol[:,:] = np.diag( (3.,3.) + (3.,) * (2*8) )
    for i in progprint_xrange(1,18):
    # for i in progprint_xrange(1,data.shape[0]):
        particlefilter.step(images[i],sideinfo=xytheta[i])

    return particlefilter, expandedpose[0]


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

# TODO this is redundant with other expand function
def expand(tracks,expandedpose):
    tracks = np.array(tracks,ndmin=3)
    expanded = np.zeros((tracks.shape[0],tracks.shape[1],expandedpose.shape[0]))

    expanded[:,:,3::3] = expandedpose[3::3]
    expanded[:,:,4] = expandedpose[4]
    expanded[:,:,5] = expandedpose[5]

    expanded[:,:,:3] = tracks[:,:,:3]
    expanded[:,:,4::3] = tracks[:,:,3::2]
    expanded[:,:,5::3] = tracks[:,:,4::2]
    return expanded

##########
#  main  #
##########

if __name__ == '__main__':
    res, expandedpose = run_randomwalk_fixednoise_sideinfo(1)
    np.save('top5tracks',expand([p.track for p in topk(res.particles,res.weights_norm,5)],expandedpose))
    np.save('meantrack',np.squeeze(expand(meantrack(res.particles,res.weights_norm),expandedpose)))
    # Neffs = np.array(res.Neff_history)
    # plt.plot(Neffs[:,0],Neffs[:,1],'bx-')
    # plt.show()

