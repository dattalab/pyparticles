from __future__ import division
import numpy as np
na = np.newaxis
import inspect, shutil, os, abc, cPickle

from renderer.renderer import MouseScene
from renderer.load_data import load_behavior_data

import pose_models
import particle_filter
import predictive_models as pm
import predictive_distributions as pd
from util.text import progprint_xrange
from util.general import joindicts

class Experiment(object):
    __metaclass__ = abc.ABCMeta

    ### must override this

    @abc.abstractmethod
    def run(self,frame_range):
        pass

    ### should probably override this

    def resume(self):
        raise NotImplementedError

    ### probably shouldn't be overridden

    def save_progress(self,particlefilter,pose_model,datapath,frame_range,**kwargs):
        dct = joindicts((
                {
                'particlefilter':particlefilter,
                'pose_model':pose_model,
                'datapath':datapath,
                'frame_range':frame_range,
                },
                kwargs
            ))

        outfilename = os.path.join(self.cachepath,str(particlefilter.numsteps))
        with open(outfilename,'w') as outfile:
            cPickle.dump(dct,outfile,protocol=2)

        descripfilename = os.path.join(self.cachepath,'description.txt')
        if not os.path.exists(descripfilename):
            with open(descripfilename,'w') as outfile:
                outfile.write(
                        '''
        experiment: %s
        data path: %s
        frame range: %s
        pose model: %s

        code.py contains the experiment code used to run the experiment; see also experiments.py

        the step-numbered files are saved with pickle via
            with open(outfilename,'w') as outfile:
                cPickle.dump((particlefilter,pose_model,datapath,frame_range),
                                outfile,protocol=2)
                        ''' % (self.__class__.__name__, datapath, frame_range, pose_model)
                        )

        shutil.copy(outfilename,os.path.join('Test Data','current_run'))

    def load_most_recent_progress(self):
        most_recent_filename = os.path.join(self.cachepath,
                max([int(x) for x in os.listdir(self.cachepath) if x.isdigit()]))
        with open(os.path.join(self.cachepath,most_recent_filename),'r') as infile:
            particlefilter, pose_model, datapath, frame_range = cPickle.load(infile)
        return particlefilter, pose_model, datapath, frame_range

    ### don't override this stuff

    def __init__(self,frame_range):
        print 'results directory: %s' % self.cachepath

        if os.path.exists(self.cachepath):
            response = raw_input('cache file exists: [o]verwrite, [r]esume, or do [N]othing? ').lower()
            if response == 'r':
                self.resume(frame_range)
            elif response == 'o':
                shutil.rmtree(self.cachepath)
            else:
                print 'did nothing'
                return

        os.makedirs(self.cachepath)

        with open(os.path.join(self.cachepath,'code.py'),'w') as outfile:
            outfile.write(inspect.getsource(self.__class__))

        self.run(frame_range)

    @property
    def cachepath(self):
        return os.path.join('results',str(abs(hash(inspect.getsource(self.__class__)))))

#################
#  Experiments  #
#################

### Decent ones

class SideInfoFixedNoise(Experiment):
    def run(self,frame_range):
        datapath = os.path.join(os.path.dirname(__file__),"Test Data","Blurred Edge")

        num_particles_firststep = 1024*50
        num_particles = 1024*30
        cutoff = 1024*15

        xytheta_noisechol = np.diag((2.,2.))

        randomwalk_noisechol = np.diag((7.,3.,0.01,2.,2.,10.,) + (20.,)*(2+2*3))
        subsequent_randomwalk_noisechol = np.diag((3.,2.,0.01,0.2,0.2,1.0,) + (6.,)*(2+2*3))

        pose_model = pose_models.PoseModel3()

        _build_mousescene(pose_model.scenefilepath)
        images, xytheta = _load_data(datapath,frame_range)

        pose_model.default_renderer_pose = pose_model.default_renderer_pose._replace(theta_yaw=xytheta[0,2])
        pose_model.default_particle_pose = pose_model.default_particle_pose._replace(theta_yaw=xytheta[0,2])

        def log_likelihood(stepnum,im,poses):
            return ms.get_likelihood(im,particle_data=pose_model.expand_poses(poses),
                x=xytheta[stepnum,0],y=xytheta[stepnum,1],theta=xytheta[stepnum,2])/500.

        pf = particle_filter.ParticleFilter(
                pose_model.particle_pose_tuple_len,
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

        pf.step(images[0],particle_kwargs={'sideinfo':xytheta[0]})
        pf.change_numparticles(num_particles)
        randomwalk_noisechol[:] = subsequent_randomwalk_noisechol[:]

        for i in progprint_xrange(1,images.shape[0],perline=10):
            if i % 10 == 0:
                self.save_progress(pf,pose_model,datapath,frame_range)
            pf.step(images[i],particle_kwargs={'sideinfo':xytheta[i]})

            print len(np.unique([p.track[1][0] for p in pf.particles]))
            print ''

        self.save_progress(pf,frame_range)


class RandomWalkFixedNoise(Experiment):
    def run(self,frame_range):
        datapath = os.path.join(os.path.dirname(__file__),"Test Data","Blurred Edge")

        num_particles_firststep = 1024*50
        num_particles = 1024*30
        cutoff = 1024*15

        randomwalk_noisechol = np.diag((3.,3.,7.,3.,0.01,2.,2.,10.,) + (20.,)*(2+2*3))
        subsequent_randomwalk_noisechol = np.diag((2.,2.,3.,2.,0.01,0.2,0.2,1.0,) + (6.,)*(2+2*3))

        pose_model = pose_models.PoseModel3()

        _build_mousescene(pose_model.scenefilepath)
        images, xytheta = _load_data(datapath,frame_range)

        pose_model.default_renderer_pose = \
            pose_model.default_renderer_pose._replace(theta_yaw=xytheta[0,2],x=xytheta[0,0],y=xytheta[0,1])
        pose_model.default_particle_pose = \
            pose_model.default_particle_pose._replace(theta_yaw=xytheta[0,2],x=xytheta[0,0],y=xytheta[0,1])

        def log_likelihood(stepnum,im,poses):
            return ms.get_likelihood(im,particle_data=pose_model.expand_poses(poses),
                x=xytheta[stepnum,0],y=xytheta[stepnum,1],theta=xytheta[stepnum,2])/500.

        pf = particle_filter.ParticleFilter(
                pose_model.particle_pose_tuple_len,
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
                self.save_progress(pf,pose_model,datapath,frame_range)
            pf.step(images[i])

            print len(np.unique([p.track[1][0] for p in pf.particles]))
            print ''

class RandomWalkFixedNoiseFrozenTrack(Experiment):
    # should look a lot like RandomWalkFixedNoise
    def run(self,frame_range):
        datapath = os.path.join(os.path.dirname(__file__),"Test Data")

        num_particles_firststep = 1024*80
        num_particles = 1024*60
        cutoff = 1024*30

        lag = 15

        randomwalk_noisechol = np.diag((3.,3.,7.,3.,0.01,2.,2.,10.,) + (20.,)*(2+2*3))
        subsequent_randomwalk_noisechol = np.diag((1.5,1.5,3.,0.25,0.01,1e-6,1e-6,1e-6,) + (5.,)*(2+2*3))
        # TODO check z size
        # TODO try cutting scale, fit on first 10 or so

        # pose_model = pose_models.PoseModel3()
        pose_model = pose_models.PoseModel10()

        _build_mousescene(pose_model.scenefilepath)
        images, xytheta = _load_data(datapath,frame_range)

        pose_model.default_renderer_pose = \
            pose_model.default_renderer_pose._replace(theta_yaw=xytheta[0,2],x=xytheta[0,0],y=xytheta[0,1])
        pose_model.default_particle_pose = \
            pose_model.default_particle_pose._replace(theta_yaw=xytheta[0,2],x=xytheta[0,0],y=xytheta[0,1])

        def log_likelihood(stepnum,im,poses):
            return ms.get_likelihood(im,particle_data=pose_model.expand_poses(poses),
                x=xytheta[stepnum,0],y=xytheta[stepnum,1],theta=xytheta[stepnum,2])/3000.

        pf = particle_filter.ParticleFilter(
                pose_model.particle_pose_tuple_len,
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

        for i in progprint_xrange(1,lag):
            pf.step(images[i])
        self.save_progress(pf,pose_model,datapath,frame_range,means=[])

        # now step with freezing means
        means = []
        for i in progprint_xrange(lag,images.shape[0],perline=10):
            means.append(np.sum(pf.weights_norm[:,na] * np.array([p.track[i-lag] for p in pf.particles]),axis=0))
            print '\nsaved a mean for index %d with %d unique particles!\n' % \
                    (i-lag,len(np.unique([p.track[i-15][0] for p in pf.particles])))

            pf.step(images[i])

            if (i % 5) == 0:
                self.save_progress(pf,pose_model,datapath,frame_range,means=means)

        self.save_progress(pf,pose_model,datapath,frame_range,means=means)

class RandomWalkFixedNoiseFrozenTrackParallel(Experiment):
    # should look a lot like RandomWalkFixedNoise
    def run(self,frame_range):
        raw_input('be sure engines are started in git root!')

        datapath = os.path.join(os.path.dirname(__file__),"Test Data")

        num_particles_firststep = 1024*80
        num_particles = 1024*60
        cutoff = 1024*30

        lag = 15

        randomwalk_noisechol = np.diag((3.,3.,7.,3.,0.01,2.,2.,10.,) + (20.,)*(2+2*3))
        subsequent_randomwalk_noisechol = np.diag((1.5,1.5,3.,0.25,0.01,1e-6,1e-6,1e-6,) + (5.,)*(2+2*3))

        # pose_model = pose_models.PoseModel3()
        pose_model = pose_models.PoseModel10()

        _build_mousescene(pose_model.scenefilepath)
        images, xytheta = _load_data(datapath,frame_range)

        pose_model.default_renderer_pose = \
            pose_model.default_renderer_pose._replace(theta_yaw=xytheta[0,2],x=xytheta[0,0],y=xytheta[0,1])
        pose_model.default_particle_pose = \
            pose_model.default_particle_pose._replace(theta_yaw=xytheta[0,2],x=xytheta[0,0],y=xytheta[0,1])

        import parallel
        # engines should be started in git root, where this file is
        dv = parallel.go_parallel(pose_model.scenefilepath,datapath,frame_range)
        def log_likelihood(stepnum,_,poses):
            dv.scatter('poses',pose_model.expand_poses(poses),block=True)
            dv.execute('''likelihoods = ms.get_likelihood(images[%d],particle_data=poses,
                                                x=xytheta[%d,0],y=xytheta[%d,1],theta=xytheta[%d,2])/2000.'''
                    % (stepnum,stepnum,stepnum,stepnum),block=True)
            return dv.gather('likelihoods',block=True)

        pf = particle_filter.ParticleFilter(
                pose_model.particle_pose_tuple_len,
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

        for i in progprint_xrange(1,lag):
            pf.step(images[i])
        self.save_progress(pf,pose_model,datapath,frame_range,means=[])

        # now step with freezing means
        means = []
        for i in progprint_xrange(lag,images.shape[0],perline=10):
            means.append(np.sum(pf.weights_norm[:,na] * np.array([p.track[i-lag] for p in pf.particles]),axis=0))
            print '\nsaved a mean for index %d with %d unique particles!\n' % \
                    (i-lag,len(np.unique([p.track[i-15][0] for p in pf.particles])))

            pf.step(images[i])

            if (i % 5) == 0:
                self.save_progress(pf,pose_model,datapath,frame_range,means=means)

        self.save_progress(pf,pose_model,datapath,frame_range,means=means)




class RandomWalkFixedNoiseFrozenTrack_AW_5Joints_simplified(Experiment):
    # should look a lot like RandomWalkFixedNoise
    def run(self,frame_range):
        datapath = os.path.join(os.path.dirname(__file__),"Test Data")

        num_particles_firststep = 1024*10
        num_particles = 1024*4
        cutoff = 1024*4

        lag = 15

        pose_model = pose_models.PoseModel_5Joint_origweights_AW()

        variances = {
            'x':             {'init':3.0,  'subsq':1.5},
            'y':             {'init':3.0,  'subsq':1.5},
            'theta_yaw':     {'init':7.0,  'subsq':3.0},
            'z':             {'init':3.0,  'subsq':0.25},
            'theta_roll':    {'init':0.01, 'subsq':0.01},
            's_w':           {'init':2.0,  'subsq':1e-6},
            's_l':           {'init':2.0,  'subsq':1e-6},
            's_h':           {'init':1.0,  'subsq':1e-6}
        }
        particle_fields = pose_model.ParticlePose._fields
        joint_names = [j for j in particle_fields if 'psi_' in j]
        [variances.update({j:{'init':20.0, 'subsq':5.0}}) for j in joint_names]

        randomwalk_noisechol = np.diag([variances[p]['init'] for p in particle_fields])
        subsequent_randomwalk_noisechol = np.diag([variances[p]['subsq'] for p in particle_fields])

        # TODO check z size
        # TODO try cutting scale, fit on first 10 or so

        

        _build_mousescene(pose_model.scenefilepath)
        images, xytheta = _load_data(datapath,frame_range)

        pose_model.default_renderer_pose = \
            pose_model.default_renderer_pose._replace(theta_yaw=xytheta[0,2],x=xytheta[0,0],y=xytheta[0,1])
        pose_model.default_particle_pose = \
            pose_model.default_particle_pose._replace(theta_yaw=xytheta[0,2],x=xytheta[0,0],y=xytheta[0,1])

        def log_likelihood(stepnum,im,poses):
            return ms.get_likelihood(im,particle_data=pose_model.expand_poses(poses),
                x=xytheta[stepnum,0],y=xytheta[stepnum,1],theta=xytheta[stepnum,2])/2000.

        pf = particle_filter.ParticleFilter(
                pose_model.particle_pose_tuple_len,
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

        for i in progprint_xrange(1,lag):
            pf.step(images[i])
        self.save_progress(pf,pose_model,datapath,frame_range,means=[])

        # now step with freezing means
        means = []
        for i in progprint_xrange(lag,images.shape[0],perline=10):
            means.append(np.sum(pf.weights_norm[:,na] * np.array([p.track[i-lag] for p in pf.particles]),axis=0))
            print '\nsaved a mean for index %d with %d unique particles!\n' % \
                    (i-lag,len(np.unique([p.track[i-15][0] for p in pf.particles])))

            pf.step(images[i])

            if (i % 5) == 0:
                self.save_progress(pf,pose_model,datapath,frame_range,means=means)

        self.save_progress(pf,pose_model,datapath,frame_range,means=means)

class RandomWalkFixedNoiseParallelSimplified(Experiment):
    # combo of previous two
    def run(self,frame_range):
        raw_input('be sure engines are started in git root!')

        datapath = os.path.join(os.path.dirname(__file__),"Test Data")

        num_particles_firststep = 1024*40
        num_particles = 1024*20
        cutoff = 1024*10

        lag = 15

        pose_model = pose_models.PoseModel_5Joint_origweights_AW()

        variances = {
            'x':             {'init':3.0,  'subsq':1.5},
            'y':             {'init':3.0,  'subsq':1.5},
            'theta_yaw':     {'init':7.0,  'subsq':3.0},
            'z':             {'init':3.0,  'subsq':0.25},
            'theta_roll':    {'init':0.01, 'subsq':0.01},
            's_w':           {'init':2.0,  'subsq':1e-6},
            's_l':           {'init':2.0,  'subsq':1e-6},
            's_h':           {'init':1.0,  'subsq':1e-6}
        }
        particle_fields = pose_model.ParticlePose._fields
        joint_names = [j for j in pose_model.ParticlePose._fields if 'psi_' in j]
        [variances.update({j:{'init':20.0, 'subsq':5.0}}) for j in joint_names]

        randomwalk_noisechol = np.diag([variances[p]['init'] for p in particle_fields])
        subsequent_randomwalk_noisechol = np.diag([variances[p]['subsq'] for p in particle_fields])

        _build_mousescene(pose_model.scenefilepath)
        images, xytheta = _load_data(datapath,frame_range)

        pose_model.default_renderer_pose = \
            pose_model.default_renderer_pose._replace(theta_yaw=xytheta[0,2],x=xytheta[0,0],y=xytheta[0,1])
        pose_model.default_particle_pose = \
            pose_model.default_particle_pose._replace(theta_yaw=xytheta[0,2],x=xytheta[0,0],y=xytheta[0,1])

        import parallel
        dv = parallel.go_parallel(pose_model.scenefilepath,datapath,frame_range)
        def log_likelihood(stepnum,_,poses):
            dv.scatter('poses',pose_model.expand_poses(poses),block=True)
            dv.execute('''likelihoods = ms.get_likelihood(images[%d],particle_data=poses,
                                                x=xytheta[%d,0],y=xytheta[%d,1],theta=xytheta[%d,2])/2500.'''
                    % (stepnum,stepnum,stepnum,stepnum),block=True)
            return dv.gather('likelihoods',block=True)

        pf = particle_filter.ParticleFilter(
                pose_model.particle_pose_tuple_len,
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

        for i in progprint_xrange(1,lag):
            pf.step(images[i])
        self.save_progress(pf,pose_model,datapath,frame_range,means=[])

        # now step with freezing means
        means = []
        for i in progprint_xrange(lag,images.shape[0],perline=10):
            means.append(np.sum(pf.weights_norm[:,na] * np.array([p.track[i-lag] for p in pf.particles]),axis=0))
            print '\nsaved a mean for index %d with %d unique particles!\n' % \
                    (i-lag,len(np.unique([p.track[i-15][0] for p in pf.particles])))

            pf.step(images[i])

            if (i % 10) == 0:
                self.save_progress(pf,pose_model,datapath,frame_range,means=means)

        self.save_progress(pf,pose_model,datapath,frame_range,means=means)


class MomentumLearnedNoiseFrozenTrackParallel(Experiment):
    # should look a lot like RandomWalkFixedNoise
    def run(self,frame_range):
        raw_input('be sure engines are started in git root!')

        datapath = os.path.join(os.path.dirname(__file__),"Test Data")

        num_particles_firststep = 1024*50
        num_particles = 1024*20
        cutoff = 1024*10

        lag = 15

        randomwalk_noisechol = np.diag((3.,3.,7.,3.,0.01,2.,2.,10.,) + (20.,)*(2+2*3))
        subsequent_randomwalk_noisechol = np.diag((1.5,1.5,3.,0.25,0.01,1e-6,1e-6,1e-6,) + (5.,)*(2+2*3))

        # pose_model = pose_models.PoseModel3()
        pose_model = pose_models.PoseModel10()

        _build_mousescene(pose_model.scenefilepath)
        images, xytheta = _load_data(datapath,frame_range)

        pose_model.default_renderer_pose = \
            pose_model.default_renderer_pose._replace(theta_yaw=xytheta[0,2],x=xytheta[0,0],y=xytheta[0,1])
        pose_model.default_particle_pose = \
            pose_model.default_particle_pose._replace(theta_yaw=xytheta[0,2],x=xytheta[0,0],y=xytheta[0,1])

        import parallel
        # engines should be started in git root, where this file is
        dv = parallel.go_parallel(pose_model.scenefilepath,datapath,frame_range)
        def log_likelihood(stepnum,_,poses):
            dv.scatter('poses',pose_model.expand_poses(poses),block=True)
            dv.execute('''likelihoods = ms.get_likelihood(images[%d],particle_data=poses,
                                                x=xytheta[%d,0],y=xytheta[%d,1],theta=xytheta[%d,2])/2000.'''
                    % (stepnum,stepnum,stepnum,stepnum),block=True)
            return dv.gather('likelihoods',block=True)

        pf = particle_filter.ParticleFilter(
                pose_model.particle_pose_tuple_len,
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
        pf.step(images[1])

        # now switch to momentum!

        starters = pf.particles

        propmatrix = np.hstack((1.5*np.eye(pose_model.particle_pose_tuple_len),-0.5*np.eye(pose_model.particle_pose_tuple_len)))
        invwishparams = (20,20*subsequent_randomwalk_noisechol)

        pf = particle_filter.ParticleFilter(
                pose_model.particle_pose_tuple_len,
                cutoff,
                log_likelihood,
                [particle_filter.AR(
                    numlags=2,
                    previous_outputs=(p.track[1],p.track[0]),
                    baseclass=lambda: pm.Momentum(propmatrix=propmatrix,noiseclass=lambda: pd.InverseWishartNoise(*invwishparams)),
                ) for p in starters]
            )

        for i in progprint_xrange(2,lag):
            pf.step(images[i])
        self.save_progress(pf,pose_model,datapath,frame_range,means=[])

        # now step with freezing means
        means = []
        for i in progprint_xrange(lag,images.shape[0],perline=10):
            means.append(np.sum(pf.weights_norm[:,na] * np.array([p.track[i-lag] for p in pf.particles]),axis=0))
            print '\nsaved a mean for index %d with %d unique particles!\n' % \
                    (i-lag,len(np.unique([p.track[i-15][0] for p in pf.particles])))

            pf.step(images[i])

            if (i % 5) == 0:
                self.save_progress(pf,pose_model,datapath,frame_range,means=means)

        self.save_progress(pf,pose_model,datapath,frame_range,means=means)


class MomentumLearnedNoiseParallelSimplified(Experiment):
    def run(self,frame_range):
        raw_input('be sure engines are started in git root!')

        datapath = os.path.join(os.path.dirname(__file__),"Test Data")

        num_particles_firststep = 1024*40
        num_particles = 1024*10
        cutoff = 1024*5

        lag = 15

        pose_model = pose_models.PoseModel_5Joint_origweights_AW()

        variances = {
            'x':             {'init':3.0,  'subsq':1.5},
            'y':             {'init':3.0,  'subsq':1.5},
            'theta_yaw':     {'init':7.0,  'subsq':3.0},
            'z':             {'init':3.0,  'subsq':0.25},
            'theta_roll':    {'init':0.01, 'subsq':0.01},
            's_w':           {'init':2.0,  'subsq':1e-6},
            's_l':           {'init':2.0,  'subsq':1e-6},
            's_h':           {'init':1.0,  'subsq':1e-6}
        }
        particle_fields = pose_model.ParticlePose._fields
        joint_names = [j for j in pose_model.ParticlePose._fields if 'psi_' in j]
        [variances.update({j:{'init':20.0, 'subsq':5.0}}) for j in joint_names]

        randomwalk_noisechol = np.diag([variances[p]['init'] for p in particle_fields])
        subsequent_randomwalk_noisechol = np.diag([variances[p]['subsq'] for p in particle_fields])

        _build_mousescene(pose_model.scenefilepath)
        images, xytheta = _load_data(datapath,frame_range)

        pose_model.default_renderer_pose = \
            pose_model.default_renderer_pose._replace(theta_yaw=xytheta[0,2],x=xytheta[0,0],y=xytheta[0,1])
        pose_model.default_particle_pose = \
            pose_model.default_particle_pose._replace(theta_yaw=xytheta[0,2],x=xytheta[0,0],y=xytheta[0,1])

        import parallel
        dv = parallel.go_parallel(pose_model.scenefilepath,datapath,frame_range)
        def log_likelihood(stepnum,_,poses):
            dv.scatter('poses',pose_model.expand_poses(poses),block=True)
            dv.execute('''likelihoods = ms.get_likelihood(images[%d],particle_data=poses,
                                                x=xytheta[%d,0],y=xytheta[%d,1],theta=xytheta[%d,2])/2500.'''
                    % (stepnum,stepnum,stepnum,stepnum),block=True)
            return dv.gather('likelihoods',block=True)

        pf = particle_filter.ParticleFilter(
                pose_model.particle_pose_tuple_len,
                cutoff,
                log_likelihood,
                [particle_filter.AR(
                    numlags=1,
                    previous_outputs=(pose_model.default_particle_pose,),
                    baseclass=lambda: pm.RandomWalk(noiseclass=lambda: pd.FixedNoise(randomwalk_noisechol))
                    ) for itr in range(num_particles_firststep)])

        pf.step(images[0])
        randomwalk_noisechol[:] = subsequent_randomwalk_noisechol[:]
        pf.step(images[1])
        pf.change_numparticles(num_particles)

        ### now switch to momentum!

        starters = pf.particles

        propmatrix = np.hstack((1.25*np.eye(pose_model.particle_pose_tuple_len),-0.25*np.eye(pose_model.particle_pose_tuple_len)))
        invwishparams = (100,100*subsequent_randomwalk_noisechol)

        pf = particle_filter.ParticleFilter(
                pose_model.particle_pose_tuple_len,
                cutoff,
                log_likelihood,
                [particle_filter.AR(
                    numlags=2,
                    previous_outputs=(p.track[1],p.track[0]),
                    baseclass=lambda: pm.Momentum(propmatrix=propmatrix,noiseclass=lambda: pd.InverseWishartNoise(*invwishparams)),
                ) for p in starters]
            )

        for i in progprint_xrange(2,lag):
            pf.step(images[i])
        self.save_progress(pf,pose_model,datapath,frame_range,means=[])

        # now step with freezing means
        means = []
        for i in progprint_xrange(lag,images.shape[0],perline=10):
            means.append(np.sum(pf.weights_norm[:,na] * np.array([p.track[i-lag] for p in pf.particles]),axis=0))
            print '\nsaved a mean for index %d with %d unique particles!\n' % \
                    (i-lag,len(np.unique([p.track[i-15][0] for p in pf.particles])))

            pf.step(images[i])

            if (i % 10) == 0:
                self.save_progress(pf,pose_model,datapath,frame_range,means=means)

        self.save_progress(pf,pose_model,datapath,frame_range,means=means)


### currently busted

class RandomWalkWithInjection(Experiment):
    def run(self,frame_range):
        datapath = os.path.join(os.path.dirname(__file__),"Test Data","Blurred Edge")

        num_particles_firststep = 1024*50
        num_particles = 1024*30
        cutoff = 1024*15

        xytheta_dart_noisechol = np.diag((2.,2.,7.))
        other_dart_noisechol = np.diag((3.,0.01,2.,2.,10.,) + (20.,)*(2+2*3))

        randomwalk_noisechol = np.diag((1.5,1.5,3.,2.,0.01,0.2,0.2,1.0,) + (6.,)*(2+2*3))

        pose_model = pose_models.PoseModel3()

        _build_mousescene(pose_model.scenefilepath)
        images, xytheta = _load_data(datapath,frame_range)

        def log_likelihood(stepnum,im,poses):
            return ms.get_likelihood(im,particle_data=pose_model.expand_poses(poses),
                x=xytheta[stepnum,0],y=xytheta[stepnum,1],theta=xytheta[stepnum,2])/3000.

        # these particles have a chance to teleport back to dart mode, which
        # should be like injection
        pf = particle_filter.ParticleFilter(
                pose_model.particle_pose_tuple_len,
                cutoff,
                log_likelihood,
                [particle_filter.AR(
                    numlags=1,
                    previous_outputs=(None,),
                    baseclass=lambda: \
                        pm.Mixture(
                            components=(
                                pm.RandomWalk(noiseclass=lambda: pd.FixedNoise(randomwalk_noisechol)),
                                pm.Concatenation(
                                    components=(
                                        pm.SideInfo(noiseclass=lambda: pd.FixedNoise(xytheta_dart_noisechol)),
                                        pm.RandomWalk(noiseclass=lambda: pd.FixedNoise(other_dart_noisechol))
                                    ),
                                    arggetters=(
                                        lambda d: {'sideinfo':d['sideinfo'][:3]},
                                        lambda d: {'lagged_outputs': [pose_model.default_particle_pose[3:],]}
                                    )
                                )
                            ),
                            pseudocounts=np.array([0.,1.]),
                            arggetters=(
                                lambda d: {'lagged_outputs':d['lagged_outputs']},
                                lambda d: {'lagged_outputs':d['lagged_outputs'],
                                            'sideinfo':d['sideinfo']}
                            )
                        )
                    ) for itr in range(num_particles_firststep)])

        # first step is special
        pf.step(images[0],particle_kwargs={'sideinfo':xytheta[0]})
        # re-calibrate after first step
        pf.change_numparticles(num_particles)
        for p in pf.particles:
            p.sampler.counts = np.array([5.,1.])*10 # TODO should this be switched? lots of teleporting? more particles? we want every history to survive for sure, so injection should be done with that in mind... each particle needs ~4k darts to its name, not randomly attached to hsitories # TODO change to proper injection, for each unique history, give it 4k darts?

        for i in progprint_xrange(1,images.shape[0],perline=10):
            # save
            if i % 10 == 0:
                self.save_progress(pf,pose_model,datapath,frame_range)

            # step
            pf.step(images[i],particle_kwargs={'sideinfo':xytheta[i]})

            # print
            print len(np.unique([p.track[1][0] for p in pf.particles]))
            print ''


class RandomWalkLearnedNoise(Experiment):
    def run(self,frame_range):
        datapath = os.path.join(os.path.dirname(__file__),"Test Data","Blurred Edge")

        num_particles_firststep = 1024*50
        num_particles = 1024*30
        cutoff = 1024*10

        initial_n_0 = 1000
        subsequent_n_0 = 16+20

        initial_randomwalk_noisecov = initial_n_0*np.diag((3.,3.,7.,3.,0.01,2.,2.,10.,) + (20.,)*(2+2*3))**2
        subsequent_randomwalk_noisecov = subsequent_n_0*np.diag((3.,3.,5.,0.5,0.01,0.05,0.05,0.5,) + (5.,)*(2+2*3))**2

        pose_model = pose_models.PoseModel3()

        _build_mousescene(pose_model.scenefilepath)
        images, xytheta = _load_data(datapath,frame_range)

        pose_model.default_renderer_pose = \
            pose_model.default_renderer_pose._replace(theta_yaw=xytheta[0,2],x=xytheta[0,0],y=xytheta[0,1])
        pose_model.default_particle_pose = \
            pose_model.default_particle_pose._replace(theta_yaw=xytheta[0,2],x=xytheta[0,0],y=xytheta[0,1])

        def log_likelihood(stepnum,im,poses):
            return ms.get_likelihood(im,particle_data=pose_model.expand_poses(poses),
                x=xytheta[stepnum,0],y=xytheta[stepnum,1],theta=xytheta[stepnum,2])/3000.

        pf = particle_filter.ParticleFilter(
                pose_model.particle_pose_tuple_len,
                cutoff,
                log_likelihood,
                [particle_filter.AR(
                    numlags=1,
                    previous_outputs=(pose_model.default_particle_pose,),
                    baseclass=lambda: pm.RandomWalk(noiseclass=lambda: pd.FixedNoise(initial_n_0,initial_randomwalk_noisecov))
                    ) for itr in range(num_particles_firststep)])

        pf.step(images[0])
        pf.change_numparticles(num_particles)
        for p in pf.particles:
            p.sampler.noisesampler.yyt[:] = 0
            p.sampler.noisesampler.S_0 = subsequent_randomwalk_noisecov
            p.sampler.noisesampler.n_n = subsequent_n_0

        for i in progprint_xrange(1,images.shape[0],perline=10):
            if i % 10 == 0:
                self.save_progress(pf,pose_model,datapath,frame_range)
            pf.step(images[i])

            print len(np.unique([p.track[1][0] for p in pf.particles]))
            print ''


class Smarticles(Experiment):
    # NOTE NOTE NOTE
    # I think this doesn't work because of autoregressive ness

    def run(self,frame_range):
        datapath = os.path.join(os.path.dirname(__file__),"Test Data","Blurred Edge")

        num_particles = 1024*20
        cutoff = 1024*10

        pose_model = pose_models.PoseModel3()

        _build_mousescene(pose_model.scenefilepath)
        images, xytheta = _load_data(datapath,frame_range)

        pose_model.default_renderer_pose = \
            pose_model.default_renderer_pose._replace(theta_yaw=xytheta[0,2],x=xytheta[0,0],y=xytheta[0,1])
        pose_model.default_particle_pose = \
            pose_model.default_particle_pose._replace(theta_yaw=xytheta[0,2],x=xytheta[0,0],y=xytheta[0,1])

        def log_likelihood(stepnum,im,poses):
            return ms.get_likelihood(im,particle_data=pose_model.expand_poses(poses),
                x=xytheta[stepnum,0],y=xytheta[stepnum,1],theta=xytheta[stepnum,2])/1000.

        ### first we do two steps of something basic to get two AR lags
        randomwalk_noisechol = np.diag((3.,3.,7.,3.,0.01,2.,2.,10.,) + (20.,)*(2+2*3))
        subsequent_randomwalk_noisechol = np.diag((1.5,1.5,3.,2.,0.01,0.2,0.2,1.0,) + (6.,)*(2+2*3))
        pf = particle_filter.ParticleFilter(
                pose_model.particle_pose_tuple_len,
                cutoff,
                log_likelihood,
                [particle_filter.AR(
                    numlags=1,
                    previous_outputs=(pose_model.default_particle_pose,),
                    baseclass=lambda: pm.RandomWalk(noiseclass=lambda: pd.FixedNoise(randomwalk_noisechol))
                    ) for itr in range(num_particles)])

        pf.step(images[0])
        randomwalk_noisechol[:] = subsequent_randomwalk_noisechol[:]
        for i in progprint_xrange(1,10,perline=10):
            pf.step(images[i])
        starter_particles = pf.particles

        ### now we build AR particles and a new particle filter

        print 'changing to AR now!'

        from numpy.lib.stride_tricks import as_strided as ast
        def AR_striding(data,nlags):
            if data.ndim == 1:
                data = np.reshape(data,(-1,1))
            sz = data.dtype.itemsize
            return ast(data,shape=(data.shape[0]-nlags,data.shape[1]*(nlags+1)),strides=(data.shape[1]*sz,sz))

        data = AR_striding(np.array(pf.particles[0].track),2)

        Syy = data[:,-16:].T.dot(data[:,-16:])
        Sytyt = data[:,:-16].T.dot(data[:,:-16])
        Syyt = data[:,-16:].T.dot(data[:,:-16])

        dof = 10
        K = Sytyt
        M = np.linalg.solve(K,Syyt.T).T.copy()
        S = Syy

        MNIWARparams = (dof,S,M,K)

        with open('mniwarparams','w') as outfile:
            cPickle.dump((MNIWARparams,pf),outfile,protocol=2)

        return

        pf = particle_filter.ParticleFilter(
                pose_model.particle_pose_tuple_len,
                cutoff,
                log_likelihood,
                [particle_filter.AR(
                    numlags=2,
                    previous_outputs=(p.track[-1],p.track[-2]),
                    baseclass=lambda: pm.HDPHSMMSampler(
                        alpha=6.,gamma=6.,
                        obs_sampler_factory=lambda: pd.MNIWAR(*MNIWARparams),
                        dur_sampler_factory=lambda: pd.Poisson(2*20,2),
                    )
                ) for p in starter_particles]
            )

        ### do a few steps with lots of particles
        for i in progprint_xrange(10,images.shape[0],perline=10):
            if i % 5 == 0:
                self.save_progress(pf,pose_model,datapath,frame_range)

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
                        numCols=msNumCols, numRows=msNumRows, useFramebuffer=True,showTiming=False)
        ms.gl_init()
    return ms

def _load_data(datapath,frame_range):
    xy = load_behavior_data(datapath,frame_range[1]+1,'centroid')[frame_range[0]:]
    theta = load_behavior_data(datapath,frame_range[1]+1,'angle')[frame_range[0]:]
    xytheta = np.concatenate((xy,theta[:,na]),axis=1)
    images = load_behavior_data(datapath,frame_range[1]+1,'images').astype('float32')[frame_range[0]:]
    images = np.array([image.T[::-1,:].astype('float32') for image in images])
    return images, xytheta

