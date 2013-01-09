from __future__ import division
import numpy as np
from pylab import *

import os, time

from IPython.parallel import Client
c = Client()
dv = c[:]

dv.execute(
'''
try:
    print ms
except:
    numCols = 32
    numRows = 32
    scenefile = "renderer/data/mouse_mesh_low_poly3.npz"
    from renderer.renderer import MouseScene
    ms = MouseScene(scenefile, mouse_width=80, mouse_height=80, \
                                scale_width = 18.0, scale_height = 200.0,
                                scale_length = 18.0, \
                                numCols=numCols, numRows=numRows, useFramebuffer=True)
    ms.gl_init()

    path_to_behavior_data="test Data/Mouse No Median Filter, No Dilation"
    from renderer.load_data import load_behavior_data
    which_img = 5
    image = load_behavior_data(path_to_behavior_data, which_img+1, 'images')[-1]
    image = image.T[::-1,:].astype('float32')
''',block=True)

def get_likelihood(particle_data):
    dv.scatter('particle_data',particle_data,block=True)
    dv.execute(
            '''
likelihoods = ms.get_likelihood(image,x=0,y=0,theta=0,particle_data=particle_data,return_posed_mice=False)
            ''',block=True)
    return dv.gather('likelihoods',block=True)

numCols = 32
numRows = 32
scenefile = os.path.join("renderer/data/mouse_mesh_low_poly3.npz")
from renderer.renderer import MouseScene
ms = MouseScene(scenefile, mouse_width=80, mouse_height=80, \
                            scale_width = 18.0, scale_height = 200.0,
                            scale_length = 18.0, \
                            numCols=numCols, numRows=numRows, useFramebuffer=True)
def main():
    num_particles = 64**2*4

    # Let's fill in our particles
    particle_data = np.zeros((num_particles, 8+ms.num_bones*3))

    # Set the horizontal offsets
    particle_data[1:,:2] = np.random.normal(loc=0, scale=1, size=(num_particles-1, 2))

    # Set the vertical offset
    particle_data[1:,2] = np.random.normal(loc=0.0, scale=3.0, size=(num_particles-1,))

    # Set the angles (yaw and roll)
    theta_val = 0
    particle_data[1:,3] = theta_val + np.random.normal(loc=0, scale=3, size=(num_particles-1,))
    particle_data[1:,4] = np.random.normal(loc=0, scale=0.01, size=(num_particles-1,))

    # Set the scales (width, length, height)
    particle_data[0,5] = np.max(ms.scale_width)
    particle_data[0,6] = np.max(ms.scale_length)
    particle_data[0,7] = np.max(ms.scale_height)
    particle_data[1:,5] = np.random.normal(loc=18, scale=2, size=(num_particles-1,))
    particle_data[1:,6] = np.random.normal(loc=18, scale=2, size=(num_particles-1,))
    particle_data[1:,7] = np.abs(np.random.normal(loc=200.0, scale=10, size=(num_particles-1,)))

    # Grab the baseline joint rotations
    rot = ms.get_joint_rotations().copy()
    rot = np.tile(rot, (int(num_particles/ms.num_mice), 1, 1))
    particle_data[:,8::3] = rot[:,:,0]
    particle_data[:,9::3] = rot[:,:,1]
    particle_data[:,10::3] = rot[:,:,2]

    # Add noise to the baseline rotations (just the pitch and yaw for now)
    # particle_data[1:,8::3] += np.random.normal(scale=20, size=(num_particles-1, ms.num_bones))
    particle_data[1:,9+6::3] += np.random.normal(scale=20, size=(num_particles-1, ms.num_bones-2))
    particle_data[1:,10::3] += np.random.normal(scale=20, size=(num_particles-1, ms.num_bones))

    # ms.gl_init()
    # path_to_behavior_data="test Data/Mouse No Median Filter, No Dilation"
    # from renderer.load_data import load_behavior_data
    # which_img = 5
    # image = load_behavior_data(path_to_behavior_data, which_img+1, 'images')[-1]
    # image = image.T[::-1,:].astype('float32')

    prev_time = time.time()

    # likelihoods = ms.get_likelihood(image, \
    #                     x=0, y=0, \
    #                     theta=0, \
    #                     particle_data=particle_data,
    #                     return_posed_mice=False)

    likelihoods = get_likelihood(particle_data)

    # np.save('parallel_likelihoods',likelihoods)

    print time.time() - prev_time


if __name__ == '__main__':
    main()
