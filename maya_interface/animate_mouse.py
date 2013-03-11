from pymel.core import *
import numpy as np
import sys, os
from collections import namedtuple
sys.path.append("/Users/Alex/Code/hsmm-particlefilters/")
os.chdir("/Users/Alex/Code/hsmm-particlefilters/")

particle_data = np.load("/Users/Alex/Dropbox/Science/Datta lab/Posture Tracking/Test Data/best joint angles/fucker.npy")

ParticlePose = namedtuple(
        'ParticlePose',
        ['x','y','theta_yaw',
         'z','theta_roll','s_w','s_l','s_h',
         'psi_z3','psi_y4','psi_z4','psi_y5'])

x = particle_data[:,0]
y = particle_data[:,1]
theta = particle_data[:,2]
z = particle_data[:,3]
scale_width = particle_data[:,5]
scale_length = particle_data[:,6]
scale_height = particle_data[:,7]

z3 = particle_data[:,-4]
y4 = particle_data[:,-3]
z4 = particle_data[:,-2]
y5 = particle_data[:,-1]


num_frames = len(z3)

cmds.playbackOptions(animationEndTime=num_frames, maxTime=num_frames)
cmds.cutKey(time=(0,num_frames))

# Unlock the translation and rotation attributes of the mouse
relevant_attributes = ["translateX", "translateY", "translateZ", "rotateY", "scaleX", "scaleY", "scaleZ"]
cmds.select("Mouse1", replace=True)
for attr in relevant_attributes:
    cmds.setAttr("Mouse1."+attr, lock=False)

default_values = {}
for attr in relevant_attributes:
    default_values[attr] = cmds.getAttr("Mouse1."+attr)

cmds.currentTime(1)
cmds.setKeyframe()

for i in range(num_frames):
    cmds.currentTime(i+2)
    
    cmds.select("joint3", replace=True)
    cmds.rotate(0, 0, z3[i], relative=False, objectSpace=True)
    cmds.setKeyframe()    
    
    cmds.select("joint4", replace=True)
    cmds.rotate(0, 0, z4[i], relative=False, objectSpace=True)
    cmds.setKeyframe()    
    cmds.rotate(0, y4[i], 0, relative=False, objectSpace=True)
    cmds.setKeyframe()    
        
    cmds.select("joint5", replace=True)
    cmds.rotate(0, y5[i], 0, relative=False, objectSpace=True)
    cmds.setKeyframe()
    
    cmds.select("Mouse1")
    cmds.setAttr("Mouse1.scaleX", scale_width[i]*default_values['scaleX'])
    cmds.setAttr("Mouse1.scaleZ", scale_length[i]*default_values['scaleZ'])
    cmds.setAttr("Mouse1.scaleY", scale_length[i]*default_values['scaleZ']) # ignore height for now
    cmds.setAttr("Mouse1.translateX", x[i]-320/2.0)
    cmds.setAttr("Mouse1.translateZ", y[i]-240/2.0)
    cmds.setKeyframe()
    
    cmds.setAttr("Mouse1.rotateY", 90+theta[i])
    cmds.setKeyframe()
    
    
#cmds.setAttr("Mouse1.scaleX", scale_x)
#cmds.setAttr("Mouse1.scaleZ", scale_z)
#cmds.setAttr("Mouse1.scaleY", scale_y)
    