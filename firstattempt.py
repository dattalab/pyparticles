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


datapath = "/Users/Alex/Dropbox/Science/Datta lab/Posture Tracking/Test Data"
# datapath = "/Users/mattjj/Dropbox/Test Data/"
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
    data = np.array([image.T[::-1,:].astype('float32') for image in data])/354.0
    # np.save('data',data)

    xy = load_behavior_data(datapath,200,'centroid')[5:]
    theta = load_behavior_data(datapath,200,'angle')[5:]
    xytheta = np.concatenate((xy,theta[:,na]),axis=1)

    # make mousescene object
    numRows, numCols = (32,32)
    num_particles = numRows*numCols*3
    ms = MouseScene(scenefilepath, mouse_width=80, mouse_height=80, \
            scale_width = 2.0, scale_height = 2.0, scale_length = 2.0\
            numCols=numCols, numRows=numRows, useFramebuffer=True,showTiming=False)
    ms.gl_init()

    rot = ms.get_joint_rotations().copy()

    # set up likelihood
    expandedpose = np.zeros((num_particles,8+3*ms.num_bones))
    expandedpose[:,8::3] = rot[0,:,0] # x angles are fixed
    def likelihood(stepnum,im,pose):
        expandedpose[:,:3] = pose[:,:3] # copy in xyz offsets
        expandedpose[:,3] = pose[:,3] # copy in theta yaw
        expandedpose[:,4] = pose[:,4] # copy in theta roll
        expandedpose[:,5:8] = pose[:,5:8] # copy in width, length and height scales
        expandedpose[:,9::3] = pose[:,3::2] # copy in y angles
        expandedpose[:,10::3] = pose[:,4::2] # copy in z angles
        # np.save('expandedpose',expandedpose)
        likelihood = ms.get_likelihood(im,particle_data=expandedpose,
                x=xytheta[stepnum,0],y=xytheta[stepnum,1],theta=xytheta[stepnum,2])
        # np.save('likelihood',likelihood)
        return likelihood

    # set up particle business
    # noisechol = np.diag( (1.,)*2 + (1.,) + (10.,)*(2*9) )
    xytheta_noisechol = np.diag( (1e-3,)*2 + (1e-3,) )
    joints_noisechol = np.diag( (1e-6,)*2 + (10.,)*(2*8) )

    initial_pose = np.zeros(3+2*9)
    initial_pose[3::2] = rot[0,:,1] # y angles
    initial_pose[4::2] = rot[0,:,2] # z angles

    initial_particles = [
            pf.AR(
                    numlags=1,
                    previous_outputs=(initial_pose,),
                    # baseclass=lambda: pm.RandomWalk(noiseclass=lambda: pd.FixedNoise(noisechol))
                    baseclass=lambda: MyModel(xytheta_noisechol, joints_noisechol)
            ) for itr in range(num_particles)]

    # create particle filter
    particlefilter = pf.ParticleFilter(8+2*ms.num_bones,cutoff,likelihood,initial_particles)

    # loop!!!
    particlefilter.step(data[0],sideinfo=xytheta[0])
    xytheta_noisechol[:,:] = np.diag( (1e-2,)*2 + (1e-2,) )
    # joints_noisechol[:,:] = np.diag( (1e-6,)*2 + (5.,) * (2*8) ) # TODO make this first step prettier
    joints_noisechol[:,:] = np.diag( (5.,) * (2*ms.num_bones) )
    for i in progprint_xrange(1,26):
    # for i in progprint_xrange(1,data.shape[0]):
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
    expanded = np.zeros((tracks.shape[0],tracks.shape[1],expandedpose.shape[0]))

    expanded[:,:,3::3] = expandedpose[3::3]
    expanded[:,:,4] = expandedpose[4]
    expanded[:,:,5] = expandedpose[5]

    expanded[:,:,:3] = tracks[:,:,:3]
    expanded[:,:,4::3] = tracks[:,:,3::2]
    expanded[:,:,5::3] = tracks[:,:,4::2]
    return expanded

ms = None
# TODO clean this thing up
def get_trackplotter(track):
    plt.interactive(True)
    track = np.array(track,ndmin=2)

    global ms
    if ms is None:
        numRows, numCols = (1,1)
        ms = MouseScene(scenefilepath, mouse_width=80, mouse_height=80, \
                scale = 2.0, \
                numCols=numCols, numRows=numRows, useFramebuffer=True, showTiming=False)
        ms.gl_init()

    xy = load_behavior_data(datapath,200,'centroid')[5:]
    theta = load_behavior_data(datapath,200,'angle')[5:]
    xytheta = np.concatenate((xy,theta[:,na]),axis=1)

    images = load_behavior_data(datapath,track.shape[0]+5,'images').astype('float32')[5:]
    images = np.array([image.T[::-1,:].astype('float32') for image in images])/354.0

    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(track[:,0],track[:,1],'b.-',label='particle track')
    plt.plot(xy[:,0],xy[:,1],'r.-',label='sideinfo track')
    plt.legend()

    def plotter(timeindex):
        plt.figure(fig.number)
        plt.subplot(2,1,2)
        ms.get_likelihood(images[timeindex],particle_data=track[na,timeindex],
                x=xytheta[timeindex,0],y=xytheta[timeindex,1],theta=xytheta[timeindex,2])
        plt.imshow(np.hstack((ms.mouse_img, ms.posed_mice[0])))

    return plotter


##########
#  main  #
##########

if __name__ == '__main__':
    res, expandedpose = run_randomwalk_fixednoise_sideinfo(500)
    np.save('top5tracks',expand([p.track for p in topk(res.particles,res.weights_norm,5)],expandedpose))
    np.save('meantrack',np.squeeze(expand(meantrack(res.particles,res.weights_norm),expandedpose)))
    # Neffs = np.array(res.Neff_history)
    # plt.plot(Neffs[:,0],Neffs[:,1],'bx-')
    # plt.show()

