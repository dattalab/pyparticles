from __future__ import division
import numpy as np
na = np.newaxis
from matplotlib import pyplot as plt
plt.interactive(True)

import predictive_models as pm
import predictive_distributions as pd
import particle_filter as pf

COLORS = ['r','g','c','m','k']

def interactive():
    nlags = 2
    MNIWARparams = (
                4,6,
                5*np.eye(2),
                np.zeros((2,2*nlags+1)),
                np.diag((1.,)*(2*nlags) + (0.01,))
                )

    particle_factory = lambda: \
            pm.AR(
                    numlags=nlags,
                    baseclass=lambda: \
                            pm.HDPHSMMSampler(
                                alpha=3.,gamma=4.,
                                obs_sampler_factory=lambda: pd.MNIWAR(*MNIWARparams),
                                dur_sampler_factory=lambda: pd.Poisson(5*10,5),
                                )
                    )


    sigma = 10.
    def loglikelihood(locs,data):
        return -np.sum((locs - data)**2,axis=1)/(2*sigma**2)

    plt.clf()

    points = [np.zeros(2)]

    particlefilter = pf.ParticleFilter(2,2500,300,loglikelihood,particle_factory)

    plt.ioff()

    # particlefilter.step(points[-1])
    # plt.plot(particlefilter.locs[:,0],particlefilter.locs[:,1],'k.')

    pts = np.array(points)
    plt.plot(pts[:,0],pts[:,1],'bo-')
    plt.xlim(-100,100)
    plt.ylim(-100,100)
    plt.draw()
    plt.ion()

    while True:
        out = plt.ginput()
        if len(out) == 0:
            break
        else:
            out = np.array(out[0])
            points.append(out)

            plt.ioff()

            plt.clf()

            particlefilter.step(out)

            for p in [particlefilter.particles[idx] for idx in np.argsort(particlefilter.weights_norm)[-5:]]:
                t = np.array(p.track)
                plt.plot(t[:,0],t[:,1],'r-')
                stateseq = np.array(p.stateseq)
                for i in range(len(set(stateseq))):
                    plt.plot(t[stateseq == i,0],t[stateseq == i,1],COLORS[i % len(COLORS)] + 'o')
                print p

            # meanpt = (particlefilter.weights_norm[:,na] * particlefilter.locs).sum(0)
            # plt.plot(meanpt[0],meanpt[1],'rx')

            # plt.plot(particlefilter.locs[:,0],particlefilter.locs[:,1],'k.')

            pts = np.array(points)
            plt.plot(pts[:,0],pts[:,1],'bo-')

            plt.xlim(-100,100)
            plt.ylim(-100,100)
            plt.draw()
            plt.ion()

    return particlefilter
