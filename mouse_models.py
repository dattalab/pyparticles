from __future__ import division
import numpy as np
na = np.newaxis

'''
The classes in this file handle different mouse models, with each class modeling
the renderer pose tuple as well as the particle pose tuple, along with the
expand_poses function to map from the particle tuple to the renderer tuple. They
also specify the default (initial) particle pose.

A Model object in this file must have the following members:

    - scenefilepath (string)
    - expanded_pose_tuple_len (int)
    - particle_pose_tuple_len (int)

    - default_particle_pose (1D array of length particle_pose_tuple_len)

    - expand_poses(particle_pose) (takes a particle pose and returns the renderer pose)
'''

# NOTE both this code and code in mousescene load the scenefile. this code loads
# it to get the default pose, which mousescene might be able to ignore in the
# future for the non-offset parameters (i.e. joint angles). mousescene loads it
# for obvious rendering purposes.

# to understand all the vector packing, see the specs at
# https://github.com/mattjj/hsmm-particlefilters/wiki/Mouse-Model-Pose-Tuple-Specs

# NOTE: in the comments below, the text <link> refers to
# https://github.com/mattjj/hsmm-particlefilters/blob/ab175f229e219f5117fde5ce76921e0014419180/renderer/renderer.py

class Model1(object):
    scenefilepath = "renderer/data/mouse_mesh_low_poly.npz"
    expanded_pose_tuple_len = 3+3*9
    particle_pose_tuple_len = 3+2*9

    def __init__(self):
        f = np.load(self.scenefilepath)
        self.joint_rotations = f['joint_rotations']

        # first three elements (x,y,theta) are ignored because of sideinfo
        self.default_particle_pose = np.zeros(self.particle_pose_tuple_len)
        self.default_particle_pose[3::2] = self.joint_rotations[:,1] # y angles
        self.default_particle_pose[4::2] = self.joint_rotations[:,2] # z angles

    def expand_poses(self,poses):
        poses = np.array(poses)
        assert poses.ndim == 2

        expandedposes = np.zeros((poses.shape[0],self.expanded_pose_tuple_len))

        expandedposes[:,:3] = poses[:,:3]                    # copy in xytheta
        expandedposes[:,3::3] = self.joint_rotations[na,:,0] # x angles
        expandedposes[:,4::3] = poses[:,3::2]                # copy in y angles
        expandedposes[:,5::3] = poses[:,4::2]                # copy in z angles

        return expandedposes


class Model2(object):
    scenefilepath = "renderer/data/mouse_mesh_low_poly2.npz"
    expanded_pose_tuple_len = 8+3*6
    particle_pose_tuple_len = 8+2*6

    def __init__(self):
        f = np.load(self.scenefilepath)
        self.joint_rotations = f['joint_rotations']

        self.default_particle_pose = np.zeros(self.particle_pose_tuple_len)

        # first three elements (x,y,theta_yaw) are ignored because of sideinfo

        # chosen from line 777 of <link>
        self.default_pose[3] = 0. # z offset

        # chosen from line 782 of <link>
        self.default_pose[4] = 0. # theta_roll

        # chosen from lines 788-790 of <link>
        self.default_particle_pose[5] = 18.  # scale_width
        self.default_particle_pose[6] = 18.  # scale_length
        self.default_particle_pose[7] = 200. # scale_height

        # joint angles from scenefile
        self.default_particle_pose[8::2] = self.joint_rotations[:,1] # y angles
        self.default_particle_pose[9::2] = self.joint_rotations[:,2] # z angles

    def expand_poses(self,poses):
        poses = np.array(poses)
        assert poses.ndim == 2

        expandedposes = np.zeros((poses.shape[0],self.expanded_pose_tuple_len))

        expandedposes[:,:2] = poses[:,:2]                    # x y
        expandedposes[:,2] = poses[:,3]                      # z
        expandedposes[:,3] = poses[:,2]                      # theta_yaw
        expandedposes[:,4] = poses[:,4]                      # copy in theta_roll
        expandedposes[:,5:8] = poses[:,5:8]                  # copy in width, length and height scales
        expandedposes[:,8::3] = self.joint_rotations[na,:,0] # x angles are fixed
        expandedposes[:,9::3] = poses[:,8::2]                # copy in y angles
        expandedposes[:,10::3] = poses[:,9::2]               # copy in z angles

        return expandedposes


class Model3(object):
    scenefilepath = "renderer/data/mouse_mesh_low_poly3.npz"
    expanded_pose_tuple_len = 8+3*5     # five joints, not six
    particle_pose_tuple_len = 7+1*2+2*3 # don't propose over theta_roll or first two y angles

    def __init__(self):
        f = np.load(self.scenefilepath)
        self.joint_rotations = f['joint_rotations']

        self.default_particle_pose = np.zeros(self.particle_pose_tuple_len)

        # first three elements (x,y,theta_yaw) are ignored because of sideinfo

        # chosen from line 777 of <link>
        self.default_pose[3] = 0. # z offset

        # chosen from lines 788-790 of <link>
        self.default_particle_pose[4] = 18.  # scale_width
        self.default_particle_pose[5] = 18.  # scale_length
        self.default_particle_pose[6] = 200. # scale_height

        # joint angles from scenefile
        # don't propose over the first two y angles, so two z's are on their own
        # at the start, then the y and z are interleaved as usual, ignoring the
        # first two y's
        self.default_particle_pose[7:9] = self.joint_rotations[:2,2]   # first two z angles
        self.default_particle_pose[9::2] = self.joint_rotations[2:,1]  # y angles
        self.default_particle_pose[10::2] = self.joint_rotations[2:,2] # z angles

    def expand_poses(self,poses):
        poses = np.array(poses)
        assert poses.ndim == 2

        expandedposes = np.zeros((poses.shape[0],self.expanded_pose_tuple_len))

        expandedposes[:,:2] = poses[:,:2]                          # x y
        expandedposes[:,2] = poses[:,3]                            # z
        expandedposes[:,3] = poses[:,2]                            # theta_yaw
        expandedposes[:,4] = 0.                                    # theta_roll is fixed
        expandedposes[:,5:8] = poses[:,5:8]                        # copy in width, length and height scales
        expandedposes[:,8::3] = self.joint_rotations[na,:,0]       # x angles are fixed
        expandedposes[:,9:9+3*2:3] = self.joint_rotations[na,:2,1] # first two y angles are fixed
        expandedposes[:,10:10+3*2:3] = poses[:,7:9]                # first two z angles are in a weird spot
        expandedposes[:,15::3] = poses[:,9::2]                     # copy in the rest of the y angles
        expandedposes[:,16::3] = poses[:,10::2]                    # copy in the rest of the z angles

        return expandedposes

