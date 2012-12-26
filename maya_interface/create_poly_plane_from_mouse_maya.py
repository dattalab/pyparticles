from pymel.core import *
import numpy as np

scale = 70.

sys.path.append("/Users/Alex/Dropbox/Science/Datta lab/Posture Tracking/hsmm-particlefilters/renderer")
path_to_behavior_data = "/Users/Alex/Dropbox/Science/Datta lab/Posture Tracking/Test Data"

from load_data import load_behavior_data


cmds.promptDialog(title="Which Image?", message="Image number:")
index = cmds.promptDialog(query=True, text=True)
index = int(index)

print index+1
images = load_behavior_data(path_to_behavior_data, index+1, "images")
image = images[-1]

width,height = image.shape

mesh_name = "mouse_mesh"
cmds.polyPlane(name=mesh_name, 
                width=width, height=height,
                subdivisionsWidth=width,
                subdivisionsHeight=height)
cmds.move(width/2., 0, height/2., relative=True)

for i in range(width*height):
    translation = cmds.xform(mesh_name +".vtx[%d]" % (i), query=True, worldSpace=True, t=True)
    x,y = translation[0], translation[2]
    x,y = int(x), int(y)
    x -= 1
    y -= 1
    translation[1] = float(image[y,x])/scale
    cmds.xform(mesh_name+".vtx[%d]"%i, worldSpace=True,t=translation)
    
cmds.move(-width/2., 0, -height/2., relative=True)
