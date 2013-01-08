from __future__ import division
import numpy as np
na = np.newaxis
from collections import namedtuple

'''
The classes in this file handle different pose models, with each class modeling
a renderer pose tuple as well as a particle pose tuple, along with the
expand_poses function to map from the particle tuple to the renderer tuple. They
also specify the default values for the full renderer pose.

The elements in RendererPose but not in ParticlePose are fixed to their
defaults, as specified in DefaultPose.

A PoseModel must have metaclass PoseModelMetaclass and should inherit from
PoseModelBase to get the default expand_poses() implementation.

A PoseModel class in this file must have the following members:

    - scenefilepath (string, path relative to git root, class member)

    - ParticlePose (namedtuple class, class member)
    - RendererPose (namedtuple class, class member)

    - default_renderer_pose (instance of RendererPose, can be instance member)

A PoseModel.__init__() method must call the superclass's __init__().

It may be a good idea to have default_renderer_pose set in __init__() so that
any errors due to filesystem state happen on object creation and not object
module load.
'''

# NOTE both this code and code for MouseScene load the scenefile. this code loads
# it to get the default pose, which mousescene might be able to ignore in the
# future. mousescene loads the scenefile for obvious rendering purposes.

##########
#  base  #
##########

# the advantages of using a metaclass are allowing a PoseModel the freedom to
# override __init__ or __new__, keeping pose specs as class members instead of
# instance members for convenient inspection, and module-load-time checking that
# the PoseModel has the right class members.

class PoseModelMetaclass(type):
    def __new__(cls,name,bases,dct):
        assert 'scenefilepath' in dct or any([hasattr(b,'scenefilepath') for b in bases])
        assert 'RendererPose' in dct or any([hasattr(b,'RendererPose') for b in bases])
        assert 'ParticlePose' in dct or any([hasattr(b,'ParticlePose') for b in bases])

        RP, PP = dct['RendererPose'], dct['ParticlePose']
        dct['_expand_indices'] = [RP._fields.index(x) for x in PP._fields]
        dct['_default_indices'] = [i for i,x in enumerate(RP._fields) if x not in PP._fields]
        dct['particle_pose_tuple_len'] = len(PP._fields)
        dct['renderer_pose_tuple_len'] = len(RP._fields)
        return super(PoseModelMetaclass,cls).__new__(cls,name,bases,dct)

class PoseModelBase(object):
    def expand_poses(self,poses):
        expanded = np.empty((poses.shape[0],self.renderer_pose_tuple_len))
        expanded[:,self._default_indices] = np.array(self.default_renderer_pose)[self._default_indices]
        expanded[:,self._expand_indices] = poses
        return expanded

    def __init__(self):
        self.default_particle_pose = self.ParticlePose(*[self.default_renderer_pose.__dict__[f]
            for f in self.ParticlePose._fields])

    # namedtuples break pickling!

    def __getstate__(self):
        return {
                '_default_indices':self._default_indices,
                '_expand_indices':self._expand_indices,
                'renderer_pose_tuple_len':self.renderer_pose_tuple_len,
                'default_renderer_pose':np.array(self.default_renderer_pose),
                }

    def __setstate__(self,dct):
        self.__dict__.update(**dct)


################
#  PoseModels  #
################

class PoseModel1(PoseModelBase):
    '''
    flattened mouse, 9 joints
    '''

    __metaclass__ = PoseModelMetaclass

    scenefilepath = "renderer/data/mouse_mesh_low_poly.npz"

    ParticlePose = namedtuple(
            'ParticlePose',
            ['x','y','theta'] + ['psi_%s%d'%(v,i) for i in range(1,10) for v in ['y','z']])

    RendererPose = namedtuple(
            'RendererPose',
            ['x','y','theta'] + ['psi_%s%d'%(v,i) for i in range(1,10) for v in ['x','y','z']])

    del i,v

    def __init__(self):
        f = np.load(self.scenefilepath)
        self.joint_rotations = jr = f['joint_rotations']

        self.default_renderer_pose = self.RendererPose(
                x=0.,y=0.,theta=0.,
                **dict(('psi_%s%d'%(v,i),jr[i-1,j]) for i in range(1,10) for j,v in enumerate(['x','y','z'])))

        super(PoseModel3,self).__init__()


class PoseModel2(PoseModelBase):
    '''
    includes theta_roll, axis scaling, and six full joints
    '''

    __metaclass__ = PoseModelMetaclass

    scenefilepath = "renderer/data/mouse_mesh_low_poly2.npz"

    ParticlePose = namedtuple(
            'ParticlePose',
            ['x','y','theta_yaw','z','theta_roll','s_w','s_l','s_h'] + \
             ['psi_%s%d'%(v,i) for i in range(1,7) for v in ['y','z']])

    RendererPose = namedtuple(
            'RendererPose',
            ['x','y','z','theta_yaw','theta_roll','s_w','s_l','s_h'] + \
             ['psi_%s%d'%(v,i) for i in range(1,7) for v in ['x','y','z']])

    del i,v

    def __init__(self):
        f = np.load(self.scenefilepath)
        self.joint_rotations = jr = f['joint_rotations']

        self.default_renderer_pose = self.RendererPose(
                x=0.,y=0.,z=0.,
                theta_yaw=0.,theta_roll=0.,
                s_w=18.,s_l=18.,s_h=200.,
                **dict(('psi_%s%d'%(v,i),jr[i-1,j]) for i in range(1,6) for j,v in enumerate(['x','y','z'])))

        super(PoseModel3,self).__init__()

class PoseModel3(PoseModelBase):
    '''
    five joints, not six as in Model2
    don't propose over theta_roll or first two joints' y angles
    '''

    __metaclass__ = PoseModelMetaclass

    scenefilepath = "renderer/data/mouse_mesh_low_poly3.npz"

    ParticlePose = namedtuple(
            'ParticlePose',
            ['x','y','theta_yaw',
             'z','theta_roll','s_w','s_l','s_h',
             'psi_z1','psi_z2',
             'psi_y3','psi_z3','psi_y4','psi_z4','psi_y5','psi_z5'])

    RendererPose = namedtuple(
            'RendererPose',
            ['x','y','z','theta_yaw','theta_roll','s_w','s_l','s_h'] + \
             ['psi_%s%d'%(v,i) for i in range(1,6) for v in ['x','y','z']])

    del i,v

    def __init__(self):
        jr = np.load(self.scenefilepath)['joint_rotations']
        self.default_renderer_pose = self.RendererPose(
                x=0.,y=0.,z=0.,
                theta_yaw=0.,theta_roll=0.,
                s_w=18.,s_l=18.,s_h=200.,
                **dict(('psi_%s%d'%(v,i),jr[i-1,j]) for i in range(1,6) for j,v in enumerate(['x','y','z'])))

        super(PoseModel3,self).__init__()


##################################
#  why are these next two here?  #
##################################

class PoseModel4(PoseModel3):
    __metaclass__ = PoseModelMetaclass

    scenefilepath = "renderer/data/mouse_mesh_low_poly3.npz"

    ParticlePose = namedtuple(
            'PoseModel3ParticlePose',
            ['x','y','theta_yaw',
             'z','theta_roll','s_w','s_l','s_h',
             'psi_z1','psi_z2',
             'psi_y3','psi_z3','psi_y4','psi_z4','psi_y5','psi_z5'])

    RendererPose = namedtuple(
            'PoseModel3RendererPose',
            ['x','y','z','theta_yaw','theta_roll','s_w','s_l','s_h'] + \
             ['psi_%s%d'%(v,i) for i in range(1,6) for v in ['x','y','z']])

    del i,v

    def __init__(self):
        jr = np.load(self.scenefilepath)['joint_rotations']
        self.default_renderer_pose = self.RendererPose(
                x=0.,y=0.,z=0.,
                theta_yaw=0.,theta_roll=0., # TODO TODO theta should be set
                s_w=18.,s_l=18.,s_h=200.,
                **dict(('psi_%s%d'%(v,i),jr[i-1,j]) for i in range(1,6) for j,v in enumerate(['x','y','z'])))

        super(PoseModel3,self).__init__()

class PoseModel5(PoseModel4):
    __metaclass__ = PoseModelMetaclass

    scenefilepath = "renderer/data/mouse_mesh_low_poly3.npz"

    ParticlePose = namedtuple(
            'PoseModel3ParticlePose',
            ['x','y','theta_yaw',
             'z','theta_roll','s_w','s_l','s_h',
             'psi_z1','psi_z2',
             'psi_y3','psi_z3','psi_y4','psi_z4','psi_y5','psi_z5'])

    RendererPose = namedtuple(
            'PoseModel3RendererPose',
            ['x','y','z','theta_yaw','theta_roll','s_w','s_l','s_h'] + \
             ['psi_%s%d'%(v,i) for i in range(1,6) for v in ['x','y','z']])

    del i,v

    def __init__(self):
        jr = np.load(self.scenefilepath)['joint_rotations']
        self.default_renderer_pose = self.RendererPose(
                x=0.,y=0.,z=0.,
                theta_yaw=0.,theta_roll=0., # TODO TODO theta should be set
                s_w=18.,s_l=18.,s_h=200.,
                **dict(('psi_%s%d'%(v,i),jr[i-1,j]) for i in range(1,6) for j,v in enumerate(['x','y','z'])))

        super(PoseModel3,self).__init__()

