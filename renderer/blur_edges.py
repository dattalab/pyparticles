from __future__ import division
from load_data import load_behavior_data
import scipy.ndimage as ndimage
from os.path import join, dirname
import numpy as np

num_frames = 8000 # total number of images in the behavior file
num_iterations_dilation = 1
num_iterations_erosion = 3
blur_sigma = (2,2)

source_behavior_data = "Test Data/Mouse No Median Filter, No Dilation"
dest_behavior_data = "/Users/Alex/Dropbox/Science/Datta lab/Posture Tracking/Test Data/Blurred Edge 2"

path_to_behavior_data = join(dirname(__file__),'..',source_behavior_data)

images = load_behavior_data(path_to_behavior_data, num_frames, 'images')

out_images = np.zeros_like(images)
for i,img in enumerate(images):
	mask = img == 0
	dilation = ndimage.morphology.binary_dilation(mask, iterations=num_iterations_dilation)
	erosion = ndimage.morphology.binary_erosion(mask, iterations=num_iterations_erosion)
	mask = dilation-erosion

	blurred_img = ndimage.filters.median_filter(img, (3,3))
	blurred_img = ndimage.gaussian_filter(blurred_img, blur_sigma)
	img[mask] = blurred_img[mask]
	out_images[i] = img

img_path = "test.int16binary"
image_height = 80
image_width = 80
out_images.tofile(join(dest_behavior_data, "Extracted Mouse Images.int16binary"))