from __future__ import division
import numpy as np
na = np.newaxis

from renderer.load_data import load_behavior_data
from renderer.renderer import MouseScene

import predictive_models as pm
import predictive_distributions as pd
import particle_filter as pf

datapath = "/Users/mattjj/Dropbox/Test Data/"
scenefilepath = "renderer/data/mouse_mesh_low_poly.npz"

# MyModel exactly the same as the following, but more explicit and maybe more efficient
#   pm.Concatenation(
#       components=(
#           pm.SideInfo(noiseclass=lambda: pd.FixedNoise(xytheta_noisechol)),
#           pm.RandomWalk(noiseclass=lambda:pd.FixedNoise(joints_noisechol))
#       ),
#       arggetters=(
#           lambda d: {'sideinfo':operator.itemgetter('sideinfo')(d)},
#           lambda d: {'lagged_outputs':
#               map(lambda x: x[2:],operator.itemgetter('lagged_outputs')(d))}
#       ))
class MyModel(object):
    def __init__(self,xytheta_noisechol,joints_noisechol):
        self.xytheta_sampler = pm.SideInfo(noiseclass=lambda: pd.FixedNoise(xytheta_noisechol))
        self.joints_sampler = pm.RandomWalk(noiseclass=lambda: pd.FixedNoise(joints_noisechol))

    def sample_next(self,sideinfo,lagged_outputs):
        return np.concatenate((
            self.xytheta_sampler.sample_next(sideinfo=sideinfo),
            self.joints_sampler.sample_next(lagged_outputs=(lagged_outputs[0][3:],))
            ))

    def copy(self):
        new = self.__new__(self.__class__)
        new.xytheta_sampler = self.xytheta_sampler.copy()
        new.joints_sampler = self.joints_sampler.copy()
        return new


def run_randomwalk_fixednoise_sideinfo(cutoff):
    # load data and sideinfo
    data = load_behavior_data(datapath,200,"images")[5:] # 680-800 also good
    data = np.array([np.rot90(i) for i in data])
    data /= 354.0

    xy = load_behavior_data(datapath,200,'centroid')[5:]
    theta = load_behavior_data(datapath,200,'angle')[5:]
    xytheta = np.concatenate((xy,theta[:,na]),axis=1)

    # make mousescene object
    numRows, numCols = (32,32)
    num_particles = numRows*numCols
    ms = MouseScene(scenefilepath, mouse_width=80, mouse_height=80, \
            scale = 2.0, \
            numCols=numCols, numRows=numRows, useFramebuffer=True)
    ms.gl_init()

    rot = ms.get_joint_rotations().copy()

    # set up likelihood
    expandedpose = np.empty((num_particles,3+3*9))
    expandedpose[:,3::3] = rot[:,:,0] # x angles are fixed
    def likelihood(im,pose):
        expandedpose[:,:3] = pose[:,:3]
        expandedpose[:,4::3] = pose[:,3::2]
        expandedpose[:,5::3] = pose[:,4::2]
        return ms.get_likelihood(im,expandedpose)

    # set up particle business
    xytheta_noisechol = np.diag( (3.,3.,3.,) )**2
    joints_noisechol = np.diag( (10.,)*(2*9) )**2

    initial_pose = np.empty(3+2*9)
    initial_pose[3::2] = rot[0,:,1]
    initial_pose[4::2] = rot[0,:,2]

    initial_particles = [
            pf.AR(
                    numlags=1,
                    previous_outputs=(initial_pose,),
                    baseclass=lambda: MyModel(xytheta_noisechol, joints_noisechol)
            ) for itr in range(num_particles)]

    # create particle filter
    particlefilter = pf.ParticleFilter(3+2*9,cutoff,likelihood,initial_particles)

    # loop!!!
    # for i in range(data.shape[0]):
    particlefilter.step(data[0],sideinfo=xytheta[0])
    joints_noisechol[np.arange(joints_noisechol.shape[0]),np.arange(joints_noisechol.shape[0])] = 5.**2
    for i in range(1,5):
        particlefilter.step(data[i],sideinfo=xytheta[i])

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

def expand(tracks,expandedpose):
    tracks = np.array(tracks,ndmin=3)
    expanded = np.empty((tracks.shape[0],tracks.shape[1],expandedpose.shape[0]))
    expanded[:,:,3::3] = expandedpose[3::3]
    expanded[:,:,:3] = tracks[:,:,:3]
    expanded[:,:,4::3] = tracks[:,:,3::2]
    expanded[:,:,5::3] = tracks[:,:,4::2]
    return expanded

##########
#  main  #
##########

if __name__ == '__main__':
    res, expandedpose = run_randomwalk_fixednoise_sideinfo(500)
    np.save('top5tracks',expand([p.track for p in topk(res.particles,res.weights_norm,5)],expandedpose))
    np.save('meantrack',np.squeeze(expand(meantrack(res.particles,res.weights_norm),expandedpose)))

