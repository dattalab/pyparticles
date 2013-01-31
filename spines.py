from os.path import *
import sys
import Joints
from collections import namedtuple
from copy import deepcopy
import numpy as np
from scipy.interpolate import Rbf, interp1d
from load_data import *

# Load in our mouse model
root_dir = "/Users/Alex/Code/hsmm-particlefilters/renderer"
scenefile = join(root_dir, "data/mouse_mesh_low_poly3.npz")
sys.path.append(root_dir)

f = np.load(scenefile)
faceNormals = f['normals']
v = f['vertices']
vertices = np.ones((len(v),4), dtype='f')
vertices[:,:3] = v
vertex_idx = f['faces']
num_vertices = vertices.shape[0]
num_indices = vertex_idx.size
joint_transforms = f['joint_transforms']
joint_weights = f['joint_weights']
joint_poses = f['joint_poses']
joint_rotations = f['joint_rotations']
joint_translations = f['joint_translations']
num_joints = len(joint_translations)

# Find the vertex with the maximum number of joints influencing it
num_joint_influences =  (joint_weights>0).sum(1).max()
num_bones = num_joints

# Load up the joints properly into a joint chain
jointChain = Joints.LinearJointChain(listOfJoints=[])
for i in range(num_bones):
    J = Joints.Joint(rotation=joint_rotations[i],\
                     translation=joint_translations[i])
    jointChain.add_joint(J)
skin = Joints.SkinnedMesh(vertices, joint_weights, jointChain)
skin.jointChain.solve_forward(0)
jointPoints = skin.jointChain.get_joint_world_positions()
vertices = skin.get_posed_vertices()

# HOW TO POSE
# ==================================================
"""
# If you want to pose some joints, go ahead!
doSomePosing = False
if doSomePosing:
    ijoint = 3
    horz_deg = 0
    vert_deg = 30
    for ijoint in [2,3,4]:
        skin.jointChain.joints[ijoint].rotation[1] = horz_deg
        skin.jointChain.joints[ijoint].rotation[2] = vert_deg
skin.jointChain.solve_forward(0)
jointPoints = skin.jointChain.get_joint_world_positions()
v = skin.get_posed_vertices()
"""

# These are the indices of mouse_mesh_low_poly3 that are down the center-spine
index = np.r_[ 19,  20,  28,  29,  30,  31,  32,  33, 123, 141, 307, 308, 309, 310, 311, 315, 428]
spine_len = vertices[index,2]
index = index[np.argsort(spine_len)]
spine_len = vertices[index,2]
spine_vert = vertices[index,1]
spine_hrz = vertices[index,0]

# Meow, we can load our best particle-filter track
best_track_file = "/Users/Alex/Dropbox/Science/Datta lab/Posture Tracking/test data/best joint angles/fucker.npy"
track = np.load(best_track_file)
num_frames, num_vars = track.shape

# The tracks in fucker.npy were calculated with the following pose model.
ParticlePose = namedtuple(
       'ParticlePose',
       ['x','y','theta_yaw',
        'z','theta_roll','s_w','s_l','s_h',
        'psi_z3','psi_y4','psi_z4','psi_y5'])

def get_vertices_from_posed_skin(track_frame, pose_model, skin):
    # Get the pose
    pose = ParticlePose(*track_frame)
    pose_dict = pose._asdict()
    # Now, parse all the fields with "psi_" and find the joint and angle
    for i,field in enumerate(pose._fields):
        if 'psi_' not in field: continue
        joint_info = field.split('psi_')[-1]
        direction = joint_info[0] # e.g. x, y or z
        which_joint = int(joint_info[1])-1 # e.g. 0,1,2,3,4
        direction = {"x":0,"y":1,"z":2}[direction] # convert from (x,y,z) to (0,1,2)
        skin.jointChain.joints[which_joint].rotation[direction] = pose_dict[field]
    skin.jointChain.solve_forward(0)
    return skin.get_posed_vertices()


# Now, go ahead and pose all those vertices
original_skin = deepcopy(skin)
vertices = []
for i in range(num_frames):
    v = get_vertices_from_posed_skin(track[i], ParticlePose, skin)
    vertices.append(v)
vertices = np.array(vertices)[:,index,:]
spine_len = vertices[:,:,2]
spine_vert = vertices[:,:,1]
spine_hrz = vertices[:,:,0]


# I'd like to show the spineogram now, please.
num_interp_pts = 100
spinogram_vert_resampled = np.zeros((num_frames, num_interp_pts))
spinogram_hrz_resampled = np.zeros((num_frames, num_interp_pts))
spinogram_vert = np.zeros((num_frames, num_interp_pts))
spinogram_hrz = np.zeros((num_frames, num_interp_pts))
bottom, top = np.min(spine_len), np.max(spine_len)
t = np.linspace(bottom, top, num_interp_pts)
for i in range(num_frames):
    spinogram_vert_resampled[i] = np.interp(t, spine_len[i], spine_vert[i], left=0, right=0)
    spinogram_hrz_resampled[i] = np.interp(t, spine_len[i], spine_hrz[i], left=0, right=0)
    this_t = np.linspace(spine_len[i][0], spine_len[i][-1], num_interp_pts)
    spinogram_vert[i] = np.interp(this_t, spine_len[i], spine_vert[i], left=0, right=0)
    spinogram_hrz[i] = np.interp(this_t, spine_len[i], spine_hrz[i], left=0, right=0)
# Now, we've gotta get the raw data spinogram
behavior_dir = "/Users/Alex/Dropbox/Science/Datta lab/Posture Tracking/test data/Mouse"
images = load_behavior_data(behavior_dir, num_frames, "images")
spine_range = np.r_[35:45]
spines = np.median(images[:,spine_range,:], axis=1)
