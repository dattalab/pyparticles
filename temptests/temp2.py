from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
plt.interactive(True)

import crp
import distributions as d

nlags = 2

a = crp.HDPHSMMARSampler(
        2,2.,3.,
        lambda: d.MNIWAR(
            2.5,1e1,
            1e1*np.eye(2),
            np.hstack((0.5*np.eye(2),np.zeros((2,(nlags-1)*2)),np.zeros((2,1)))),
            np.diag((10.,)*(2*nlags) + (10.,))),
        lambda: d.Poisson(5*10,5)
        )

data = []
for itr in range(100):
    data.append(a.sample_next())
    if np.linalg.norm(data[-1]) > 500:
        break
data = np.array(data)

plt.figure()
plt.plot(data[:,0],data[:,1],'k-')
stateseq = np.array(a.stateseq)
for i in range(len(set(stateseq))):
    plt.plot(data[stateseq == i,0],data[stateseq == i,1],'o')

