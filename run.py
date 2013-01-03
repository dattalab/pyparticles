from __future__ import division
import numpy as np
na = np.newaxis
from warnings import warn
import os

from renderer.load_data import load_behavior_data
from renderer.renderer import MouseScene

import particle_filter as pf
from util.text import progprint_xrange

############################
#  computation parameters  #
############################

msNumRows, msNumCols = (32,32)
num_particles = msNumRows*msNumCols*5

####################
#  global objects  #
####################

# these global objects are shared for several purposes and convenient for
# interactive access. making them lazily loaded provides fast importing

ms = None # MouseScene object, global so that it's only built once
xytheta = images = None # offset sideinfo sequence and image sequence

def _build_mousescene(conf):
    scenefilepath = os.path.join(os.path.dirname(__file__),conf.pose_model.scenefilepath)

    global ms
    if ms is None:
        ms = MouseScene(scenefilepath, mouse_width=80, mouse_height=80, \
                numCols=msNumCols, numRows=msNumRows, useFramebuffer=True,showTiming=False)
        ms.gl_init()
    return ms

# TODO this could be hidden in experiment_configurations.py
def _load_data_and_sideinfo(conf):
    datapath = os.path.join(os.path.dirname(__file__),conf.datapath)
    frame_range = conf.frame_range

    global xytheta, images
    if xytheta is None or images is None:
        xy = load_behavior_data(datapath,sum(frame_range),'centroid')[frame_range[0]:]
        theta = load_behavior_data(datapath,sum(frame_range),'angle')[frame_range[0]:]
        xytheta = np.concatenate((xy,theta[:,na]),axis=1)
        images = load_behavior_data(datapath,sum(frame_range),'images').astype('float32')[frame_range[0]:]
        images = np.array([image.T[::-1,:].astype('float32') for image in images])/354.0
    return xytheta, images

#############
#  running  #
#############

def run(conf,num_particles,cutoff,num_particles_firststep):
    # load 3D model object, data, sideinfo
    ms = _build_mousescene(conf)
    xytheta, images = _load_data_and_sideinfo(conf)

    # build the particle filter
    particlefilter = pf.ParticleFilter(
                        conf.pose_model.particle_pose_tuple_len,
                        cutoff,
                        conf.get_log_likelihood(ms,xytheta),
                        conf.get_initial_particles(num_particles_firststep))

    # first step is special
    particlefilter = pf.step(images[0],sideinfo=xytheta[0])
    particlefilter.change_numparticles(num_particles)
    conf.first_step_done(particlefilter)

    # run the other steps
    for i in progprint_xrange(1,images.shape[0]):
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

def render(conf,stepnum,poses):
    warn('untested')
    # might be slow; needs to do a full mousescene render pass
    _build_mousescene(), _load_data_and_sideinfo()
    return ms.get_likelihood(np.zeros((msNumRows,msNumCols)),particle_data=conf.pose_model.expand_poses(poses),
            x=xytheta[stepnum,0],y=xytheta[stepnum,1],theta=xytheta[stepnum,2],
            return_posed_mice=True)[1]

##########
#  main  #
##########

if __name__ == '__main__':
    raise NotImplementedError # TODO

