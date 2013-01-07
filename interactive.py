from __future__ import division
import numpy as np
na = np.newaxis
from matplotlib import pyplot as plt
plt.interactive(True)

import predictive_models as pm
import predictive_distributions as pd
import particle_filter as pf

COLORS = ['r','g','c','m','k']

# TODO plot mean path in gray!

##########################
#  experiment functions  #
##########################

def smart():
    raise NotImplementedError
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

    def plotfunc(particles,weights):
        for p in topk(particles,weights,5):
            t = np.array(p.track)
            plt.plot(t[:,0],t[:,1],'r-')
            stateseq = np.array(p.stateseq)
            for i in range(len(set(stateseq))):
                plt.plot(t[stateseq == i,0],t[stateseq == i,1],COLORS[i % len(COLORS)] + 'o')
            print p

    return interactive(2500,500,particle_factory,plotfunc)

def dumb_momentum_fixednoise():
    raise NotImplementedError
    propmatrix = np.hstack((2*np.eye(2),-1*np.eye(2)))
    noisechol = 20*np.eye(2)
    particle_factory = lambda: \
            pm.AR(
                    numlags=2,
                    initial_obs=[np.zeros(2) for itr in range(2)],
                    baseclass=lambda: \
                            pm.Momentum(
                                propmatrix=propmatrix,
                                noiseclass=lambda: pd.FixedNoise(noisechol=noisechol))
                    )

    def plotfunc(particles,weights):
        plottopk(particles,weights,5)
        plotmeanpath(particles,weights)

    return interactive(5000,2500,particle_factory,plotfunc)


def dumb_momentum_learnednoise():
    raise NotImplementedError
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

    def plotfunc(particles,weights):
        plottopk(particles,weights,5)

    return interactive(5000,2500,particle_factory,plotfunc)



def dumb_randomwalk_fixednoise():
    noisechol = 30*np.eye(2)
    initial_particles = [
            pf.AR(
                    numlags=1,
                    previous_outputs=[np.zeros(2)],
                    baseclass=lambda: \
                        pm.RandomWalk(noiseclass=lambda: pd.FixedNoise(noisechol=noisechol))
                    ) for itr in range(10000)]

    def plotfunc(particles,weights):
        plottopk(particles,weights,5)
        plotmeanpath(particles,weights)

    return interactive(initial_particles,2500,plotfunc)

def dumb_randomwalk_learnednoise():
    num_pseudoobs = 25
    noisecov = 30**2*np.eye(2) * num_pseudoobs
    initial_particles = [
            pf.AR(
                    numlags=1,
                    previous_outputs=[np.zeros(2)],
                    baseclass=lambda: \
                        pm.RandomWalk(noiseclass=lambda: pd.InverseWishartNoise(num_pseudoobs,noisecov))
                    ) for itr in range(10000)]

    def plotfunc(particles,weights):
        plottopk(particles,weights,5)
        plotmeanpath(particles,weights)

    return interactive(initial_particles,2500,plotfunc)


def interactive(initial_particles,cutoff,plotfunc):
    sigma = 10.
    def loglikelihood(_,locs,data):
        return -np.sum((locs - data)**2,axis=1)/(2*sigma**2)

    plt.clf()

    points = [np.zeros(2)]

    particlefilter = pf.ParticleFilter(2,cutoff,loglikelihood,initial_particles)

    plt.ioff()

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
            particlefilter.change_numparticles(5000) # TESTING

            plotfunc(particlefilter.particles,particlefilter.weights_norm)

            pts = np.array(points)
            plt.plot(pts[:,0],pts[:,1],'bo--')

            plt.xlim(-100,100)
            plt.ylim(-100,100)
            plt.draw()
            plt.ion()

    return particlefilter

###########
#  utils  #
###########

def topk(items,scores,k):
    return [items[idx] for idx in np.argsort(scores)[:-(k+1):-1]]

def plottopk(particles,weights,k):
    for p in topk(particles,weights,k):
        t = np.array(p.track)
        plt.plot(t[:,0],t[:,1],'rx-')
        print p

def plotmeanpath(particles,weights):
    track = np.array(particles[0].track)*weights[0,na]
    for p,w in zip(particles[1:],weights[1:]):
        track += np.array(p.track) * w
    plt.plot(track[:,0],track[:,1],'k^:')

