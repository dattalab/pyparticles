from __future__ import division
import numpy as np

import Image
import sys
from os.path import *
from renderer.load_data import load_behavior_data

# TODO make this nicer

# TODO add an angle plotting function

num_frames = 5+70
max_vert = 500
dest_dir = "/Users/mattjj/Desktop/movie_new"

# Load the real data
path_to_behavior_data = "Test Data"
imgs = load_behavior_data(path_to_behavior_data, num_frames+1, "images")[5:]
centroids = load_behavior_data(path_to_behavior_data, num_frames+1, "centroid")[5:]
x,y = centroids[:,0], centroids[:,1]
theta = load_behavior_data(path_to_behavior_data, num_frames+1, "angle")[5:]

# Load the synthetic data
posed_mice = np.load("posed_mice.npy")
for i in range(len(posed_mice)):
	posed_mice[i] = posed_mice[i][::-1].T
tracks = np.load("pftrack.npy")
x_synth = tracks[:,0]
y_synth = tracks[:,1]
theta_synth = tracks[:,2]

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


# Make a movie of the real stuff
for i in range(num_frames-5):
	I_real = embed_image(imgs[i], x[i], y[i], theta[i], (240,320))
	I_real = np.clip(I_real, 0, max_vert)
	I_real = (I_real.astype('float32')/max_vert)*255.0
	
	I_synth = embed_image(posed_mice[i], x_synth[i], y_synth[i], theta_synth[i], (240,320))
	I_synth = np.clip(I_synth, 0, max_vert)
	I_synth = (I_synth.astype('float32')/max_vert)*255.0

	I = np.hstack((I_real, I_synth))

	Image.fromarray(I.astype('uint8')).save(join(dest_dir, "%03d.png" % i))



# TODO
def movie(conf,track):#,outdir):
    # import Image
    # from util.general import scoreatpercentile
    _build_mousescene(conf), _load_data_and_sideinfo(conf)

    # _d = images.flatten()
    # scale = scoreatpercentile(_d[_d != 0],90,0)[0]

    rendered_images = ms.get_likelihood(
            np.zeros(images[0].shape),
            particle_data=conf.pose_model.expand_poses(track),
            x=xytheta[:,0],
            y=xytheta[:,1],
            theta=xytheta[:,2],
            return_posed_mice=True)[1]

    np.save('posed_mice.npy',rendered_images)

    # for i, (truth, rendered) in progprint(enumerate(zip(images,rendered_images)),total=len(rendered_images)):
        # Image.fromarray(np.clip(np.hstack((truth,rendered))*(255./scale),0,255.).astype('uint8'))\
        #         .save(os.path.join(outdir,'frame%d.png'%(i+conf.frame_range[0])))
