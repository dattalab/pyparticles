from __future__ import division
import numpy as np
import Image, cPickle, os

from renderer.load_data import load_behavior_data
from renderer.renderer import MouseScene

import particle_filter

max_vert = 500
dest_dir = '/Users/mattjj/Desktop/movie_new'

def meantrack_movie(pf_file):
    with open(pf_file,'r') as infile:
        pf, pose_model, datapath, frame_range = cPickle.load(infile)
    track = particle_filter.meantrack(pf)
    return movie(track,pose_model,datapath,frame_range)

def movie(track,pose_model,datapath,frame_range):
    images, xytheta = _load_data(datapath,(frame_range[0],frame_range[0]+track.shape[0]))

    ms = _build_mousescene(pose_model.scenefilepath)
    posed_mice = ms.get_likelihood(
            np.zeros(images[0].shape),
            particle_data=pose_model.expand_poses(track),
            x=xytheta[:,0],
            y=xytheta[:,1],
            theta=xytheta[:,2],
            return_posed_mice=True)[1]


    for i in range(len(posed_mice)):
        posed_mice[i] = posed_mice[i][::-1].T

    x_synth = track[:,0]
    y_synth = track[:,1]
    theta_synth = track[:,2]

    for i in range(images.shape[0]):
        I_real = embed_image(images[i], xytheta[i,0], xytheta[i,1], xytheta[i,2], (240,320))
        I_real = np.clip(I_real, 0, max_vert)
        I_real = (I_real.astype('float32')/max_vert)*255.0

        I_synth = embed_image(posed_mice[i], x_synth[i], y_synth[i], theta_synth[i], (240,320))
        I_synth = np.clip(I_synth, 0, max_vert)
        I_synth = (I_synth.astype('float32')/max_vert)*255.0

        I = np.hstack((I_real, I_synth))

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
    images = np.array([image.T[::-1,:].astype('float32') for image in images])
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
