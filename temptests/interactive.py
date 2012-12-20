from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
plt.interactive(True)

import predictive_models
import predictive_distributions as d

nlags = 2
MNIWARparams = (
            2.5,1e1,
            1e1*np.eye(2),
            np.hstack((0.5*np.eye(2),np.zeros((2,(nlags-1)*2)),np.zeros((2,1)))),
            np.diag((10.,)*(2*nlags) + (10.,))
            )

a = predictive_models.AR(
        numlags=2,
        baseclass=lambda: \
            predictive_models.HDPHSMMSampler(
                alpha=2.,gamma=3.,
                obs_sampler_factory=lambda: d.MNIWAR(*MNIWARparams),
                dur_sampler_factory=lambda: d.Poisson(5*10,5),
                )
        )
