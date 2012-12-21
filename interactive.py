from __future__ import division
import numpy as np
na = np.newaxis
from matplotlib import pyplot as plt
plt.interactive(True)

import predictive_models as pm
import predictive_distributions as pd
import particle_filter as pf

COLORS = ['r','g','c','m','k']

def smart():
    nlags = 2
    MNIWARparams = (
                3,
                10*np.eye(2),
                np.zeros((2,2*nlags+1)),
                np.diag((10,)*(2*nlags) + (0.1,))
                )

    particle_factory = lambda: \
            pm.AR(
                    numlags=nlags,
                    initial_obs=[np.zeros(2) for itr in range(nlags)],
                    baseclass=lambda: \
                            pm.HDPHSMMSampler(
                                alpha=3.,gamma=4.,
                                obs_sampler_factory=lambda: pd.MNIWAR(*MNIWARparams),
                                dur_sampler_factory=lambda: pd.Poisson(4*5,5),
                                )
                    )

    def plotfunc(p):
        t = np.array(p.track)
        plt.plot(t[:,0],t[:,1],'r-')
        stateseq = np.array(p.stateseq)
        for i in range(len(set(stateseq))):
            plt.plot(t[stateseq == i,0],t[stateseq == i,1],COLORS[i % len(COLORS)] + 'o')
        print p

    interactive(2500,500,particle_factory,plotfunc)


def dumb_noise():
    particle_factory = lambda: \
            pm.AR(
                    numlags=1,
                    initial_obs=[np.zeros(2)],
                    baseclass=lambda: \
                        pm.RandomWalk(noiseclass=lambda: pd.InverseWishartNoise(10,10*30*np.eye(2)))
                    )

    def plotfunc(p):
        t = np.array(p.track)
        plt.plot(t[:,0],t[:,1],'rx-')
        print p

    interactive(5000,2000,particle_factory,plotfunc)


def dumb_momentum():
    propmatrix = np.hstack((2*np.eye(2),-1*np.eye(2)))
    invwishparams = (10,10.*30*np.eye(2))
    particle_factory = lambda: \
            pm.AR(
                    numlags=2,
                    initial_obs=[np.zeros(2) for itr in range(2)],
                    baseclass=lambda: \
                        pm.Momentum(
                            propmatrix=propmatrix,
                            noiseclass=lambda: pd.InverseWishartNoise(*invwishparams))
                    )

    def plotfunc(p):
        t = np.array(p.track)
        plt.plot(t[:,0],t[:,1],'rx-')
        print p

    interactive(10000,2000,particle_factory,plotfunc)


def interactive(nparticles,cutoff,particle_factory,plotfunc):
    sigma = 10.
    def loglikelihood(locs,data):
        return -np.sum((locs - data)**2,axis=1)/(2*sigma**2)

    plt.clf()

    points = [np.zeros(2)]

    particlefilter = pf.ParticleFilter(2,nparticles,cutoff,loglikelihood,particle_factory)

    plt.ioff()

    pts = np.array(points)
    plt.plot(pts[:,0],pts[:,1],'bx-')
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
                plotfunc(p)

            pts = np.array(points)
            plt.plot(pts[:,0],pts[:,1],'bo-')

            plt.xlim(-100,100)
            plt.ylim(-100,100)
            plt.draw()
            plt.ion()

    return particlefilter

# TODO synthetic


