from __future__ import division
import numpy as np
np.set_printoptions(linewidth=100)
from matplotlib import pyplot as plt
plt.interactive(True)

import distributions as d
from collections import deque
import util.stats as stats

ndims = 2
nlags = 2

params = (
                ndims+0.5,
                np.eye(ndims),
                np.hstack((
                    # AR
                    0.9*np.eye(ndims), np.zeros((ndims,(nlags-1)*ndims)),
                    # drift
                    np.zeros((ndims,1))
                    )),
                np.diag( (10.,)*(ndims*nlags) + (1.,) )
        )

while True:
    allobs = []
    lags = deque(maxlen=nlags)
    a = d.MNIWAR(*params)

    for itr in range(20):
        x = a.sample_next(lags)
        allobs.append(x)
        lags.appendleft(x)
        if np.linalg.norm(x) > 400:
            break

    allobs = np.array(allobs)
    plt.plot(allobs[:,0],allobs[:,1],'x-')

    print stats.sample_mniw(a.n,a.sigma_n,a.M_n,np.linalg.inv(a.K_n))
    plt.draw()

    if len(plt.ginput()) == 0:
        break
    else:
        plt.clf()

