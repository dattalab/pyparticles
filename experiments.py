from __future__ import division
import numpy as np
na = np.newaxis
import inspect, shutil, os, abc, cPickle

from renderer.load_data import load_behavior_data
from renderer.renderer import MouseScene

import pose_models
import particle_filter
import predictive_models as pm
import predictive_distributions as pd
from util.text import progprint_xrange

class Experiment(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def run(self):
        pass

    ### should probably override this

    def resume(self):
        raise NotImplementedError

    ### common but could be overridden

    def save_progress(self,particlefilter,frame_range):
        self._cache_setup()
        outfilename = os.path.join(self.cachepath,str(particlefilter.numsteps))
        with open(outfilename,'w') as outfile:
            cPickle.dump((particlefilter,frame_range),outfile,protocol=2)
        shutil.copy(outfilename,'current_run')

    def load_most_recent_progress(self):
        most_recent_filename = os.path.join(self.cachepath,
                max([int(x) for x in os.listdir(self.cachepath) if x.isdigit()]))
        with open(os.path.join(self.cachepath,most_recent_filename),'r') as infile:
            particlefilter, frame_range = cPickle.load(infile)
        return particlefilter, frame_range

    ### don't override this stuff

    def _cache_setup(self):
        if not hasattr(self,'_cache_is_set_up'):
            if os.path.exists(self.cachepath):
                if raw_input('overwrite existing data? ').lower() != 'y':
                    print 'did nothing'
                    return
                else:
                    shutil.rmtree(self.cachepath)
            else:
                os.makedirs(self.cachepath)

            with open(os.path.join(self.cachepath,'code')) as outfile:
                outfile.write(inspect.getsource(self.__class__))

            self._cache_is_set_up = True

    @property
    def cachepath(self):
        return hash(inspect.getsource(self.__class__))

#################
#  Experiments  #
#################


class SideInfoFixedNoise(Experiment):
    def run(self):
        datapath = os.path.join(os.path.dirname(__file__),"Test Data","Blurred Edge")
        frame_range = (5,1000)

        num_particles_firststep = 1024*50
        num_particles = 1024*20
        cutoff = 1024*10

        xytheta_noisechol = np.diag((2.,2.))

        randomwalk_noisechol = np.diag((7.,3.,0.01,2.,2.,10.,) + (20.,)*(2+2*3))
        subsequent_randomwalk_noisechol = np.diag((3.,2.,0.01,0.2,0.2,1.0,) + (6.,)*(2+2*3))

        pose_model = pose_models.PoseModel3()

        _build_mousescene(pose_model.scenefilepath)
        images, xytheta = _load_data(datapath,frame_range)

        pose_model.default_renderer_pose = pose_model.default_renderer_pose._replace(theta_yaw=xytheta[0,2])
        pose_model.default_particle_pose = pose_model.default_particle_pose._replace(theta_yaw=xytheta[0,2])

        def log_likelihood(stepnum,im,poses):
            return ms.get_likelihood(im,particle_data=self.pose_model.expand_poses(poses),
                x=xytheta[stepnum,0],y=xytheta[stepnum,1],theta=xytheta[stepnum,2])/500.

        pf = particle_filter.ParticleFilter(
                pose_model.pose_tuple_len,
                cutoff,
                log_likelihood,
                [particle_filter.AR(
                    numlags=1,
                    previous_outputs=(pose_model.default_particle_pose,),
                    baseclass=lambda: \
                            pm.Concatenation(
                                components=(
                                    pm.SideInfo(noiseclass=lambda: pd.FixedNoise(xytheta_noisechol)),
                                    pm.RandomWalk(noiseclass=lambda: pd.FixedNoise(randomwalk_noisechol))
                                ),
                                arggetters=(
                                    lambda d: {'sideinfo':d['sideinfo'][:2]},
                                    lambda d: {'lagged_outputs': map(lambda x: x[2:],d['lagged_outputs'])}
                                )
                            )
                    ) for itr in range(num_particles_firststep)])

        pf.step(images[0],sideinfo=xytheta[0])
        pf.change_numparticles(num_particles)
        randomwalk_noisechol[:] = subsequent_randomwalk_noisechol[:]

        for i in progprint_xrange(1,images.shape[0],perline=10):
            if i % 10 == 0:
                self.save_progress(pf,frame_range)
            pf.step(images[i],sideinfo=xytheta[i])

        self.save_progress(pf,frame_range)


class RandomWalkFixedNoise(Experiment):
    def run(self):
        datapath = os.path.join(os.path.dirname(__file__),"Test Data","Blurred Edge")
        frame_range = (5,1000)

        num_particles_firststep = 1024*50
        num_particles = 1024*20
        cutoff = 1024*10

        randomwalk_noisechol = np.diag((3.,3.,7.,3.,0.01,2.,2.,10.,) + (20.,)*(2+2*3))
        subsequent_randomwalk_noisechol = np.diag((2.,2.,3.,0.5,0.01,0.1,0.1,0.5,) + (5.,)*(2+2*3))

        pose_model = pose_models.PoseModel3()

        _build_mousescene(pose_model.scenefilepath)
        images, xytheta = _load_data(datapath,frame_range)

        pose_model.default_renderer_pose = \
            pose_model.default_renderer_pose._replace(theta_yaw=xytheta[0,2],x=xytheta[0,0],y=xytheta[0,1])
        pose_model.default_particle_pose = \
            pose_model.default_particle_pose._replace(theta_yaw=xytheta[0,2],x=xytheta[0,0],y=xytheta[0,1])

        def log_likelihood(stepnum,im,poses):
            return ms.get_likelihood(im,particle_data=self.pose_model.expand_poses(poses),
                x=xytheta[stepnum,0],y=xytheta[stepnum,1],theta=xytheta[stepnum,2])/500.

        pf = particle_filter.ParticleFilter(
                pose_model.pose_tuple_len,
                cutoff,
                log_likelihood,
                [particle_filter.AR(
                    numlags=1,
                    previous_outputs=(pose_model.default_particle_pose,),
                    baseclass=lambda: pm.RandomWalk(noiseclass=lambda: pd.FixedNoise(randomwalk_noisechol))
                    ) for itr in range(num_particles_firststep)])

        pf.step(images[0])
        pf.change_numparticles(num_particles)
        randomwalk_noisechol[:] = subsequent_randomwalk_noisechol[:]

        for i in progprint_xrange(1,images.shape[0],perline=10):
            if i % 10 == 0:
                self.save_progress(pf,frame_range)
            pf.step(images[i])

######################
#  Common Utilities  #
######################

msNumRows, msNumCols = 32,32
ms = None
def _build_mousescene(scenefilepath):
    global ms
    if ms is None:
        ms = MouseScene(scenefilepath, mouse_width=80, mouse_height=80, \
                        scale_width = 18.0, scale_height = 200.0,
                        scale_length = 18.0, \
                        numCols=msNumCols, numRows=msNumRows, useFramebuffer=False,showTiming=False)
        ms.gl_init()
    return ms

def _load_data(datapath,frame_range):
    xy = load_behavior_data(datapath,frame_range[1]+1,'centroid')[frame_range[0]:]
    theta = load_behavior_data(datapath,frame_range[1]+1,'angle')[frame_range[0]:]
    xytheta = np.concatenate((xy,theta[:,na]),axis=1)
    images = load_behavior_data(datapath,frame_range[1]+1,'images').astype('float32')[frame_range[0]:]
    images = np.array([image.T[::-1,:].astype('float32') for image in images])
    return images, xytheta
