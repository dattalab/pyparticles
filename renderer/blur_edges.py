from __future__ import division
from load_data import load_behavior_data
import scipy.ndimage as ndimage
from os.path import join, dirname
import numpy as np

num_frames = 8000 # total number of images in the behavior file
num_iterations_dilation = 1
blur_sigma = (2,2)

source_behavior_data = "../Test Data/Mouse No Median Filter, No Dilation"
dest_behavior_data = "../Test Data/Blurred Edge"

path_to_behavior_data = join(dirname(__file__),'..',source_behavior_data)

images = load_behavior_data(path_to_behavior_data, num_frames, 'images')

out_images = np.zeros_like(images)
for i,img in enumerate(images):
	mask = img == 0
	dilation = ndimage.morphology.binary_dilation(mask, iterations=num_iterations_dilation)
	mask = dilation-mask

	blurred_img = ndimage.filters.median_filter(img, (5,5))
	img[mask] = blurred_img[mask]
	out_images[i] = img

out_images.tofile(join(dest_behavior_data, "Extracted Mouse Images.int16binary"))