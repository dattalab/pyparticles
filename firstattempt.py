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
    expandedpose = np.ones((num_particles,3+3*9)) * -1111
    expandedpose[:,3::3] = rot[:,:,0] # x angles are fixed
    expandedpose[:,4] = rot[:,0,1] # as are first tuple of y and z
    expandedpose[:,5] = rot[:,0,2]
    def likelihood(im,pose):
        expandedpose[:,:3] = pose[:,:3]
        expandedpose[:,7::3] = pose[:,3::2] # y angles
        expandedpose[:,8::3] = pose[:,4::2] # z angles
        return ms.get_likelihood(im,expandedpose)

    # set up particle business
    # xytheta_noisechol = np.diag( (3.,3.,3.,) )
    # joints_noisechol = np.diag( (10.,)*(2*8) )

    noisechol = np.diag( (1.,)*2 + (5.,) + (10.,)*(2*8) )

    initial_pose = np.zeros(3+2*8)
    initial_pose[3::2] = rot[0,1:,1] # y angles
    initial_pose[4::2] = rot[0,1:,2] # z angles

    np.save('initialpose',initial_pose) # TODO remove
    np.save('rot',rot)

    initial_particles = [
            pf.AR(
                    numlags=1,
                    previous_outputs=(initial_pose,),
                    baseclass=lambda: pm.RandomWalk(noiseclass=lambda: pd.FixedNoise(noisechol))
            ) for itr in range(num_particles)]

    # create particle filter
    particlefilter = pf.ParticleFilter(3+2*8,cutoff,likelihood,initial_particles)

    # loop!!!
    particlefilter.step(data[0])
    noisechol[3:,3:] = np.diag( (10.,) * (2*8) )
    for i in progprint_xrange(1,3):
    # for i in progprint_xrange(1,data.shape[0]):
        particlefilter.step(data[i])

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
    expanded = np.ones((tracks.shape[0],tracks.shape[1],expandedpose.shape[0]))*-2222

    expanded[:,:,3::3] = expandedpose[3::3]
    expanded[:,:,4] = expandedpose[4]
    expanded[:,:,5] = expandedpose[5]

    expanded[:,:,:3] = tracks[:,:,:3]
    expanded[:,:,7::3] = tracks[:,:,3::2]
    expanded[:,:,8::3] = tracks[:,:,4::2]
    return expanded

ms = None
def get_trackplotter(track):
    global ms
    if ms is None:
        numRows, numCols = (1,1)
        ms = MouseScene(scenefilepath, mouse_width=80, mouse_height=80, \
                scale = 2.0, \
                numCols=numCols, numRows=numRows, useFramebuffer=True)
        ms.gl_init()

    images = load_behavior_data(datapath,track.shape[0]+5,'images').astype('float32')[5:]/354.0
    images = np.array([np.rot90(i) for i in images])

    def plotter(timeindex):
        plt.interactive(True)
        ms.get_likelihood(images[timeindex],track[na,timeindex])
        plt.imshow(np.hstack((ms.mouse_img, ms.posed_mice[0])))

    return plotter


##########
#  main  #
##########

if __name__ == '__main__':
    res, expandedpose = run_randomwalk_fixednoise_sideinfo(500)
    np.save('top5tracks',expand([p.track for p in topk(res.particles,res.weights_norm,5)],expandedpose))
    np.save('meantrack',np.squeeze(expand(meantrack(res.particles,res.weights_norm),expandedpose)))

