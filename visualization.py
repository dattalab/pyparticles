import Image
import sys
from os.path import *
sys.path.append("/Users/Alex/Dropbox/Science/Datta lab/Posture Tracking/hsmm-particlefilters/renderer/")
from load_data import load_behavior_data

num_frames = 1000
max_vert = 500
dest_dir = "/Users/Alex/Desktop/Movie/Real Data"
dest_synth_dir = "/Users/Alex/Desktop/Movie/Synth Data"

# Load the real data
path_to_behavior_data = "/Users/Alex/Dropbox/Science/Datta lab/Posture Tracking/Test Data"
imgs = load_behavior_data(path_to_behavior_data, num_frames, "images")[4:]
centroids = load_behavior_data(path_to_behavior_data, num_frames, "centroid")[4:]
x,y = centroids[:,0], centroids[:,1]
theta = load_behavior_data(path_to_behavior_data, num_frames, "angle")[4:]

# Load the synthetic data
posed_mice = np.load("/Users/Alex/Dropbox/Science/Datta lab/Posture Tracking/Test Data/first_results/posed_mice.npy")
for i in range(len(posed_mice)):
	posed_mice[i] = posed_mice[i][::-1].T
tracks = np.load("/Users/Alex/Dropbox/Science/Datta lab/Posture Tracking/Test Data/first_results/tracks.npy")
x_synth = tracks[0,:,0]
y_synth = tracks[0,:,1]
theta_synth = tracks[0,:,2]

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


