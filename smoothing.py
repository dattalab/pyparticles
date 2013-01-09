import numpy as np
import visualization, pose_models

from scipy.signal import gaussian

a = gaussian(5*2,2)

a/=a.sum()

oldtrack = np.load('tracks.npy')[0]

smoothedall = np.array([np.convolve(col,a,'same') for col in oldtrack.T]).T.copy()

visualization.movie(smoothedall,pose_models.PoseModel3(),'Test Data',(5,1000))
