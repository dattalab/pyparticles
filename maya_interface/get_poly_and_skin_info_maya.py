from pymel.core import *
import numpy as np

# Select the mesh we care about
mesh_name = "skin"
out_name = "mouse_mesh_low_poly"
cmds.select(mesh_name)

# Get number of faces, number of vertices
numVertices = cmds.polyEvaluate(vertex=True)
numFaces = cmds.polyEvaluate(face=True)

# Get vertices
xOrig = cmds.xform(mesh_name+".vtx[*]", query=True, worldSpace=True, t=True)
vertices = zip(xOrig[0::3], xOrig[1::3], xOrig[2::3])

# Get per-face vertex indices
strFaces = cmds.polyInfo(faceToVertex=True)
faces = []
for face in strFaces:
    while face.find('  ') != -1:
        face = face.replace('  ', ' ')
    face = face.split(' ')
    face = [int(v) for v in face[2:-1]]
    faces.append(face)

# Get per-face normals
strNormals = cmds.polyInfo(faceNormals=True)
normals = []
for normal in strNormals:
    while normal.find('  ') != -1:
        normal= normal.replace('  ', ' ')
    normal = normal.replace('\n', '')
    normal = normal.split(' ')
    normal = [float(v) for v in normal[2:]]
    normals.append(normal)

normals = np.vstack(normals)
vertices = np.vstack(vertices)
faces = np.vstack(faces) # will fail if mesh is not triangulated

# Select the mesh we care about
clusters = cmds.ls(type='skinCluster')
skinCluster_name = clusters[0]

# Get the influenced geometry (just a name)
geometry = cmds.skinCluster(skinCluster_name, query=True, geometry=True)
geometry_name = geometry[0]

# Get the transforms that influence the geometry (the names of the joints)
influences = cmds.skinCluster(skinCluster_name, query=True, weightedInfluence=True)

# Get the influence weights for each joint
joint_weights = []
num_vertices = cmds.polyEvaluate(geometry_name, vertex=True)
for i in range(num_vertices):
    these_weights = cmds.skinPercent(skinCluster_name, \
                                    geometry_name+".vtx[%d]"%i, \
                                    query=True, value=True)
    joint_weights.append(these_weights)
joint_weights = np.vstack(joint_weights)


joint_transforms = np.zeros((len(influences), 4,4))
joint_poses = np.zeros((len(influences), 4,4))
rotations = np.zeros((len(influences), 3))
translations = np.zeros_like(rotations)
for i,influence in enumerate(influences):
    this_transform = np.asarray(cmds.xform(influence, matrix=True, query=True, worldSpace=True))
    joint_transforms[i] = this_transform.reshape(4,4, order='C')

    this_pose = np.asarray( cmds.getAttr("%s.bindPose"%influence) )
    joint_poses[i] = this_pose.reshape(4,4, order='C')

    rotations[i] = np.asarray( cmds.getAttr("%s.jointOrient"%influence) )
    translations[i] = cmds.xform(influence, translation=True, query=True)
    print rotations[i]
    


np.savez("/Users/Alex/Dropbox/Science/Datta lab/Posture Tracking/Models/%s.npz" % out_name, \
                            normals=normals, vertices=vertices, faces=faces, \
                            joint_weights=joint_weights, joint_transforms=joint_transforms, \
                            joint_poses=joint_poses,\
                            joint_rotations=rotations, joint_translations=translations)