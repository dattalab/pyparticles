from __future__ import division
import numpy as np
na = np.newaxis
from collections import namedtuple
import operator, abc

from util.general import joindicts

'''
The classes in this file handle different mouse models, with each class modeling
the renderer pose tuple as well as the particle pose tuple, along with the
expand_poses function to map from the particle tuple to the renderer tuple. They
also specify the default values for the full renderer pose.

The elements in RendererPose but not in ParticlePose are fixed to their
defaults, as specified in DefaultPose.

A Model object in this file must have the following members:

    - scenefilepath (string)
    - renderer_pose_tuple_len (int)
    - particle_pose_tuple_len (int)

    - expand_poses(particle_pose) (takes a particle pose and returns the renderer pose)

    these three are recommended but not necessary
    they are required for the default expand_poses()
    - ParticlePose (namedtuple class)
    - RendererPose (namedtuple class)
    - DefaultPose (instance of RendererPose)
'''

# NOTE both this code and code in mousescene load the scenefile. this code loads
# it to get the default pose, which mousescene might be able to ignore in the
# future for the non-offset parameters (i.e. joint angles). mousescene loads it
# for obvious rendering purposes.

# to understand all the vector packing, see the specs at
# https://github.com/mattjj/hsmm-particlefilters/wiki/Mouse-Model-Pose-Tuple-Specs

# NOTE: in the comments below, the text <link> refers to
# https://github.com/mattjj/hsmm-particlefilters/blob/ab175f229e219f5117fde5ce76921e0014419180/renderer/renderer.py

def _flatten(lst):
    reduce(operator.add,lst)


class MouseModelABC(object):
    __metaclass__ = abc.ABCMeta

    def expand_poses(self,poses):
        # this default version can be overridden if it is too slow
        return np.array([self.DefaultPose.replace(**self.ParticlePose(*pose).__dict__) for pose in poses])

class MouseModel1(object):
    scenefilepath = "renderer/data/mouse_mesh_low_poly.npz"
    renderer_pose_tuple_len = 3+3*9
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


class MouseModel2(object):
    scenefilepath = "renderer/data/mouse_mesh_low_poly2.npz"
    renderer_pose_tuple_len = 8+3*6
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
        expandedposes[:,5:8] = np.abs(poses[:,5:8])          # copy in width, length and height scales
        expandedposes[:,8::3] = self.joint_rotations[na,:,0] # x angles are fixed
        expandedposes[:,9::3] = poses[:,8::2]                # copy in y angles
        expandedposes[:,10::3] = poses[:,9::2]               # copy in z angles

        return expandedposes


class MouseModel3(MouseModelABC):
    '''
    five joints, not six as in Model2
    don't propose over theta_roll or first two joints' y angles
    '''

    scenefilepath = "renderer/data/mouse_mesh_low_poly3.npz"

    ParticlePose = namedtuple(
            'Model3ParticlePose',
            ['x','y','theta_yaw',
             'z','theta_roll','s_w','s_l','s_h',
             'psi_z1','psi_z2','psi_y3','psi_z3','psi_y4','psi_z4','psi_y5','psi_z5']
            )

    RendererPose = namedtuple(
            'Model3RendererPose',
            ['x','y','z','theta_yaw','theta_roll','s_w','s_l','s_h',
             'psi_y1','psi_z1','psi_y2','psi_z2','psi_y3','psi_z3','psi_y4','psi_z4','psi_y5','psi_z5']
             #['psi_%s%d'%(v,i) for i in range(1,6) for v in ['x','y','z']]
            )

    particle_pose_tuple_len = len(ParticlePose._fields)
    renderer_pose_tuple_len = len(RendererPose._fields)

    def __init__(self):
        # construct DefaultPose for each instance so that any errors only happen on
        # class instantiation, not on module load
        f = np.load(self.scenefilepath)
        self.joint_rotations = jr = f['joint_rotations']

        self.DefaultValues = self.RendererPose(
                x=0.,y=0.,z=0.,
                theta_yaw=0.,theta_roll=0.,
                s_w=18.,s_l=18.,s_h=200.,
                **joindicts((
                             dict(('psi_x%d'%i,jr[i,0]) for i in range(1,6)),
                             dict(('psi_y%d'%i,jr[i,1]) for i in range(1,6)),
                             dict(('psi_z%d'%i,jr[i,2]) for i in range(1,6)),
                            )))

