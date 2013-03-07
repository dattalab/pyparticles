from __future__ import division
import os
import numpy as np
na = np.newaxis
import Image, cPickle, os, shutil

from renderer.load_data import load_behavior_data
from renderer.renderer import MouseScene

import particle_filter

max_vert = 500
# dest_dir = '/Users/Alex/Desktop/movies'
# dest_dir = '/Users/mattjj/Desktop/movie_new'
dest_dir = "/home/dattalab/movies"
dest_dir2 = "/home/dattalab/movies"
# dest_dir2 = '/Users/mattjj/Desktop/sidebyside_movie_new/'

# Whenever you see "pf_file", it is operating on the named pickled files in e.g. 
# results/3245470323142455654.6335/101
def frozentrack_movie(pf_file,offset=0):

    with open(pf_file,'r') as infile:
        it = cPickle.load(infile)

        if isinstance(it,tuple):
            # old version
            pf, pose_model, datapath, frame_range = cPickle.load(infile)
        elif isinstance(it,dict):
            pf = it['particlefilter']
            pose_model = it['pose_model']
            datapath = it['datapath']
            frame_range = it['frame_range']
            means = it['means']

    track = np.array(means)
    # return movie_sidebyside(track,pose_model,datapath,frame_range)
    return movie_sidebyside_cuda(track,pose_model,datapath,frame_range)
    # return movie(track,pose_model,datapath,frame_range,offset=offset)

def meantrack_movie(pf_file):
    with open(pf_file,'r') as infile:
        it = cPickle.load(infile)

        if isinstance(it,tuple):
            # old version
            pf, pose_model, datapath, frame_range = cPickle.load(infile)
        elif isinstance(it,dict):
            pf = it['particlefilter']
            pose_model = it['pose_model']
            datapath = it['datapath']
            frame_range = it['frame_range']

    track = particle_filter.meantrack(pf)
    return movie(track,pose_model,datapath,frame_range)

def movie_sidebyside(track,pose_model,datapath,frame_range):
    images, xytheta = _load_data(datapath,(frame_range[0],frame_range[0]+track.shape[0]-1))

    track2 = track.copy()
    track2[:,:2] = 0

    ms = _build_mousescene(pose_model.scenefilepath)
    posed_mice = ms.get_likelihood(
            np.zeros(images[0].shape),
            particle_data=pose_model.expand_poses(track2),
            x=0,
            y=0,
            theta=xytheta[:,2],
            return_posed_mice=True)[1]

    scaling = posed_mice.max()

    for i in range(len(posed_mice)):
        Image.fromarray((np.hstack((images[i][:,::-1].T,posed_mice[i]))/scaling*255.0).astype('uint8')).save(os.path.join(dest_dir2, "%03d.png" % i))
    

def movie_sidebyside_cuda(track,pose_model,datapath,frame_range):
    import pymouse
    from MouseData import MouseData
    from MousePoser import MousePoser

    # images, xytheta = _load_data(datapath,(frame_range[0],frame_range[0]+track.shape[0]-1))

    m = MouseData(scenefile=os.path.abspath(pose_model.scenefilepath))
    mp = MousePoser(mouseModel=m, maxNumBlocks=10, imageSize=(64,64))

    # Load in our real data, extracted from the Kinect
    mm = pymouse.Mousemodel(datapath, 
                            n=np.max(frame_range),
                            image_size=(mp.resolutionY,mp.resolutionX))
    mm.load_data()
    mm.clean_data(normalize_images = False, filter_data=True)
    images = mm.images[frame_range[0]:]

    # Find the nearest multiple of numMicePerPass, and pad the track
    numMiceToPose = int(np.ceil(len(track)/float(mp.numMicePerPass))*mp.numMicePerPass)
    track2 = np.zeros((numMiceToPose, pose_model.particle_pose_tuple_len), dtype='float32')
    track2[:len(track)] = track.copy()


    # ==============================

    joint_angles = mp.baseJointRotations_cpu + pose_model.get_joint_rotations(track2)
    scales = pose_model.get_scales(track2)
    offsets = pose_model.get_offsets(track2)
    rotations = pose_model.get_rotations(track2)

    numPasses = int(np.ceil(numMiceToPose / mp.numMicePerPass))
    posed_mice = np.zeros((numPasses*mp.numMicePerPass,mp.resolutionY, mp.resolutionX), dtype='float32')
    for i in range(numPasses):
        start = i*mp.numMicePerPass
        end = start+mp.numMicePerPass
        l,p = mp.get_likelihoods(joint_angles=joint_angles[start:end], \
                                scales=scales[start:end], \
                                offsets=offsets[start:end], \
                                rotations=rotations[start:end], \
                                real_mouse_image=None, \
                                save_poses=True)
        posed_mice[i*mp.numMicePerPass:i*mp.numMicePerPass+mp.numMicePerPass] = p

    # ==============================

    scaling = max(posed_mice.max(), images.max())

    for i in range(len(track)):
        Image.fromarray((np.hstack((images[i][:,::-1].T,posed_mice[i]))/scaling*255.0).astype('uint8')).save(os.path.join(dest_dir2, "%03d.png" % i))


def movie(track,pose_model,datapath,frame_range,offset=0):
    images, xytheta = _load_data(datapath,(frame_range[0],frame_range[0]+track.shape[0]-1))

    track2 = track.copy()
    track2[:,:2] = 0

    ms = _build_mousescene(pose_model.scenefilepath)
    posed_mice = ms.get_likelihood(
            np.zeros(images[0].shape),
            particle_data=pose_model.expand_poses(track2),
            x=0,
            y=0,
            theta=xytheta[:,2],
            return_posed_mice=True)[1]

    for i in range(len(posed_mice)):
        posed_mice[i] = posed_mice[i][::-1].T

    x_synth = track[:,0]
    y_synth = track[:,1]
    theta_synth = track[:,2]

    for i in range(offset,images.shape[0]):
        I_real = embed_image(images[i], xytheta[i,0], xytheta[i,1], xytheta[i,2], (240,320))
        I_real = np.clip(I_real, 0, max_vert)
        I_real = (I_real.astype('float32')/max_vert)*255.0

        I_synth = embed_image(posed_mice[i], x_synth[i], y_synth[i], theta_synth[i], (240,320))
        I_synth = np.clip(I_synth, 0, max_vert)
        I_synth = (I_synth.astype('float32')/max_vert)*255.0

        I = np.hstack((I_real, I_synth))

        # shutil.rmtree(dest_dir)
        # os.makedirs(dest_dir)
        Image.fromarray(I.astype('uint8')).save(os.path.join(dest_dir, "%03d.png" % i))

######################
#  Common Utilities  #
######################

msNumRows, msNumCols = 32,32
ms = None
def _build_mousescene(scenefilepath):
    global ms
    if ms is None:
        ms = MouseScene(scenefilepath, mouse_width=80, mouse_height=80, \
                        scale_width = 18.0, scale_height = 200.0,
                        scale_length = 18.0, \
                        numCols=msNumCols, numRows=msNumRows, useFramebuffer=True,showTiming=False)
        ms.gl_init()
    else:
        assert scenefilepath == ms.scenefile, 'restart your interpreter!'
    return ms

def _load_data(datapath,frame_range):
    xy = load_behavior_data(datapath,frame_range[1]+1,'centroid')[frame_range[0]:]
    theta = load_behavior_data(datapath,frame_range[1]+1,'angle')[frame_range[0]:]
    xytheta = np.concatenate((xy,theta[:,na]),axis=1)
    images = load_behavior_data(datapath,frame_range[1]+1,'images').astype('float32')[frame_range[0]:]
    return images, xytheta


def embed_image(img, x, y, theta, large_img_size=(240,320)):
    import scipy.ndimage.interpolation as interp
    h,w = img.shape
    img = interp.rotate(img.copy(), theta, mode='constant')
    large_img = np.zeros(large_img_size, dtype=img.dtype)
    large_img[:img.shape[0],:img.shape[1]] = img
    offsetx = x - img.shape[1]/2.0
    offsety = y - img.shape[0]/2.0
    large_img = interp.shift(large_img, (offsety, offsetx))
    return large_img
