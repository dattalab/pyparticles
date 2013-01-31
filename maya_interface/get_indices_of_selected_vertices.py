from pymel.core import *
import numpy as np
 
sv = cmds.ls(selection=True, shortNames=True)
 
i = 0
idx = np.array([], dtype='int32')
 
for i in range(len(sv)):
    left = sv[i].find('[')+1
    right = sv[i].find(']')
    idx_text = sv[i][left:right]
    these_vertices = eval('np.r_[{0}]'.format(idx_text))
    idx = np.hstack((idx, these_vertices))
 
print idx
 
# FOR "low poly mouse no arms 1-20-2013.ma" is:
# [ 19  20  28  29  30  31  32  33 123 141 307 308 309 310 311 315 428]
