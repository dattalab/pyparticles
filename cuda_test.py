import sys
import os

from collections import namedtuple
import numpy as np

import pymouse
from poser.MouseData import MouseData
from poser.MousePoser import MousePoser

from experiments import Experiment
from pose_models import PoseModelBase, PoseModelMetaclass
import pose_models
import particle_filter
import predictive_models as pm
import predictive_distributions as pd
from util.text import progprint_xrange
from util.general import joindicts
import inspect

import mongowrapper

class RandomWalkFixedNoiseCUDA(Experiment):
    # Here's the run function.
    def run(self,frame_range):

        db = mongowrapper.MongoWrapper(db_name = "behavior", 
                        collection_name = "particle_filter", 
                        hostname="washington.hms.harvard.edu", 
                        username="dattalab", 
                        password="I8freecheese")

        # First things first, we have to figure out where our mouse image data is coming from
        datapath = os.path.abspath(os.path.join(os.path.dirname(__file__),"data"))

        # This helps smooth tracks.
        lag = 15

        # This pose_model defines how many joints we are allowed to move.
        # The pose_model is allowed to restrict the set of joints that the particle
        # filter can move. Check out the methods and classes there.
        pose_model = pose_models.PoseModel_5Joints_XZ_forCUDA()

        particle_fields = pose_model.ParticlePose._fields
        joint_names = [j for j in particle_fields if 'psi_' in j]

        # The variances here define how widespread our guesses are, based off the previous guess.
        variances = {
            'x':             {'init':3.0,  'subsq':1.5},
            'y':             {'init':3.0,  'subsq':1.5},
            'theta_yaw':     {'init':7.0,  'subsq':3.0},
            'z':             {'init':3.0,  'subsq':0.25},
            'theta_roll':    {'init':0.01, 'subsq':0.01},
            's_w':           {'init':0.025,  'subsq':1e-6},
            's_l':           {'init':0.025,  'subsq':1e-6},
            's_h':           {'init':1.0,  'subsq':1e-6}
        }        
        [variances.update({j:{'init':20.0, 'subsq':5.0}}) for j in joint_names]

        limits = {
            'x':             (-10,10),
            'y':             (-10,10),
            'theta_yaw':     (-20, 20),
            'z':             (-20,30),
            'theta_roll':    (-10,10),
            's_w':           (0.001, 1),
            's_l':           (0.15, 0.25),
            's_h':           (0.5, 500),
            'psi_z1':        (-50,30),
            'psi_z2':        (-30,30),
            'psi_z3':        (-60,60),
            'psi_z4':        (-60,60),
            'psi_z5':        (-60,60),
            'psi_x1':        (-15,15),
            'psi_x2':        (-15,15),
            'psi_x3':        (-45,45),
            'psi_x4':        (-60,60),
            'psi_x5':        (-45, 45),

        }

        minmaxpairs = [limits[k] for k in particle_fields]

        randomwalk_noisechol = np.diag([variances[p]['init'] for p in particle_fields])
        subsequent_randomwalk_noisechol = np.diag([variances[p]['subsq'] for p in particle_fields])

        # TODO check z size
        # TODO try cutting scale, fit on first 10 or so

        # This builds our likelihood model
        # MouseData contains information about the synthetic mouse (joint angles, skin vertices, etc)
        # MousePoser contains methods to pose a synthetic mouse
        m = MouseData(scenefile=os.path.abspath(pose_model.scenefilepath))
        mp = MousePoser(mouseModel=m, maxNumBlocks=10, maxNumThreads=256, imageSize=(64,64))

        # Then, we define how many particles we want to run
        # (particles are higher for the first step to start with a good guess)
        # numMicePerPass = 2560 or something, usually
        num_particles_firststep = mp.numMicePerPass*20
        num_particles = mp.numMicePerPass*5
        cutoff = mp.numMicePerPass*2

        # Load in our real data, extracted from the Kinect
        mm = pymouse.Mousemodel(datapath, 
                                n=np.max(frame_range),
                                image_size=(mp.resolutionY,mp.resolutionX))
        mm.load_data()
        mm.clean_data(normalize_images=False, filter_data=True)

        # This is a tiny tiny hack, where we use our knowledge of where we THINK
        # the mouse is to start off our particle filter. 
        theta, x, y = mm.angle[frame_range[0]:], mm.centroid[frame_range[0]:,0], mm.centroid[frame_range[0]:,1]
        images = mm.images[frame_range[0]:]

        pose_model.default_renderer_pose = \
            pose_model.default_renderer_pose._replace(theta_yaw=theta[0],x=0,y=0)
        pose_model.default_particle_pose = \
            pose_model.default_particle_pose._replace(theta_yaw=theta[0],x=0,y=0)

        # We pre-allocate the array to save the proposed particles. It's a huge array, but whatever, whatever, whatever.
        num_regular_steps = images.shape[0] - lag
        particle_data = np.zeros((num_regular_steps, pose_model.particle_pose_tuple_len), dtype='float32')


        # This is an interface between the CUDA poser/rasterizer and the particle filter,
        # that helps translate how good each guess is between the two classes.
        # Its main function is to make sure that we convert from "particle poses"
        # to "renderer poses". This is only important because we don't propose
        # over every degree of freedom in our model. 
        def log_likelihood(stepnum,im,poses):
            joint_angles = mp.baseJointRotations_cpu + pose_model.get_joint_rotations(poses)
            scales = pose_model.get_scales(poses)

            # The X and Y scales are not allowed to be out of [0,1],
            # because that inevitably causes memory access issues on the card.
            # This is an open issue #4
            scales[:,0] = np.clip(scales[:,0], 0, 1)
            scales[:,1] = np.clip(scales[:,1], 0, 1)
            offsets = pose_model.get_offsets(poses)
            rotations = pose_model.get_rotations(poses)

            assert np.mod(len(joint_angles), mp.numMicePerPass) == 0, \
                "Number of particles must be a multiple of the number of mice per pass, %d" % mp.numMicePerPass

            numPasses = len(joint_angles) / mp.numMicePerPass
            likelihoods = np.zeros((numPasses*mp.numMicePerPass,), dtype='float32')
            posed_mice = np.zeros((numPasses*mp.numMicePerPass,mp.resolutionY, mp.resolutionX), dtype='float32') # FOR DEBUGGING ONLY

            for i in range(numPasses):
                start = i*mp.numMicePerPass
                end = start+mp.numMicePerPass
                l,p = mp.get_likelihoods(joint_angles=joint_angles[start:end], \
                                        scales=scales[start:end], \
                                        offsets=offsets[start:end], \
                                        rotations=rotations[start:end], \
                                        real_mouse_image=im[:,::-1].T.astype('float32'), \
                                        save_poses=True)
    
                likelihoods[i*mp.numMicePerPass:i*mp.numMicePerPass+mp.numMicePerPass] = l
                posed_mice[i*mp.numMicePerPass:i*mp.numMicePerPass+mp.numMicePerPass] = p

            idx = np.argsort(likelihoods)[::-1]
            q = posed_mice[idx[0]]
            r = posed_mice[idx[1]]
            s = posed_mice[idx[2]]
            q = np.hstack((im[:,::-1].T, q, r, s))
            q = 254.0*q/q.max()
            import Image
            Image.fromarray(q.astype('uint8')).save("/home/dattalab/poses/%d.png" % stepnum)

            return likelihoods / 2000.0


        # Okay, initialize the particle filter (we're using a random walk particle filter)
        pf = particle_filter.ParticleFilter(
                pose_model.particle_pose_tuple_len,
                cutoff,
                log_likelihood,
                [particle_filter.LimitedAR(
                    minmaxpairs=minmaxpairs,
                    numlags=1,
                    previous_outputs=(pose_model.default_particle_pose,),
                    baseclass=lambda: pm.RandomWalk(noiseclass=lambda: pd.FixedNoise(randomwalk_noisechol))
                    ) for itr in range(num_particles_firststep)])


        # Do the first frame (we make a TON of guesses to get a good starting point)
        pf.step(images[0])

        # Okay, now we start off the particle filter (which uses lags for smoothing)
        # by running #lags frames in to the future. This "prebuffers" our particle filter.
        for i in progprint_xrange(1,lag):
            pf.step(images[i])

        # Now, we dial down the number of guesses, and the variance of the guesses
        # for subsequent frames. 
        pf.change_numparticles(num_particles)
        randomwalk_noisechol[:] = subsequent_randomwalk_noisechol[:]

        # TODO: REWRITE SAVE_PROGRESS TO USE MONGODB STUFF
        # self.save_progress(pf,pose_model,datapath,frame_range,means=[])

        # Now, this is the bulk of the work. Run through all frames, saving the means
        # at every 5 frames
        for i in progprint_xrange(lag,images.shape[0],perline=10):
            particle_data[i-lag] = np.sum(pf.weights_norm[:,np.newaxis] * np.array([p.track[i-lag] for p in pf.particles]), axis=0)

            print '\nsaved a mean for index %d with %d unique particles!\n' % \
                    (i-lag,len(np.unique([p.track[i-15][0] for p in pf.particles])))

            pf.step(images[i])

            if (i % 10) == 0:
                # self.save_progress(pf,pose_model,datapath,frame_range,means=particle_data)
                progress_dict = {"experiment_id": self.cachepath(frame_range),
                    "iteration":i,
                    "particle_means":particle_data,
                    "weights":pf.log_weights,
                    "num_effective_particles":np.asarray(pf._Neff_history),
                    "num_surviving_particles":np.asarray(pf._Nsurvive_history),
                    "num_particles":len(pf.particles),
                    "proportion_surviving":float(pf._Nsurvive_history[-1][1])/len(pf.particles),
                    "best_likelihood":pf.log_weights.max(),
                    "frame_range":frame_range,
                    "datapath":datapath,
                    "code": inspect.getsource(self.__class__)}
                db.save(progress_dict)

        # Save everything out once we're done. The means are the most important part right now!!
        # self.save_progress(pf,pose_model,datapath,frame_range,means=particle_data)
        progress_dict = {"experiment_id": self.cachepath(frame_range),
            "iteration":i,
            "particle_means":particle_data,
            "weights":pf.log_weights,
            "num_effective_particles":np.asarray(pf._Neff_history),
            "num_surviving_particles":np.asarray(pf._Nsurvive_history),
            "num_particles":len(pf.particles),
            "proportion_surviving":float(pf._Nsurvive_history[-1][1])/len(pf.particles),
            "best_likelihood":pf.log_weights.max(),
            "frame_range":frame_range,
            "datapath":datapath,
            "code": inspect.getsource(self.__class__),
            "completed":True}
        db.save(progress_dict)



RandomWalkFixedNoiseCUDA((45,1000))