from pymel.core import *
import numpy as np

filename = "/Users/Alex/Dropbox/Science/Datta lab/Posture Tracking/Models/ExemplarImage.npz"
f = np.load(filename)
image = f['image']

width,height = image.shape

mesh_name = "mouse_mesh"
cmds.polyPlane(name=mesh_name, 
                width=10, height=10,
                subdivisionsWidth=width,
                subdivisionsHeight=height)

for i in range(width*height):
    translation = cmds.xform(mesh_name +".vtx[%d]" % i, query=True, worldSpace=True, t=True)
    translation[1] = float(image.ravel()[i])/700.
    cmds.xform(mesh_name+".vtx[%d]"%i, worldSpace=True,t=translation)
