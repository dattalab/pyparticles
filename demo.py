from renderer.renderer import MouseScene
from renderer.load_data import load_behavior_data

import pose_models
import particle_filter
import predictive_models as pm
import predictive_distributions as pd
from util.text import progprint_xrange

import numpy as np
na = np.newaxis

# ==================================================
#                     SERIAL
# ==================================================
# This is how we do things serially
import os
import experiments
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

experiments.RandomWalkFixedNoiseFrozenTrack_AW_5Joints_simplified((5,100)).run((5,100))
# This saves data into results/{HASH}/
# where HASH is randomly generated for that experiment (see experiments.cachepath())
# and in results/{HASH}/ are numbered files.
# These files are pickled dictionaries. 
# So, for instance, on my laptop, I have the following path:
# /Users/Alex/Code/hsmm-particlefilters/results/1834870540556565875
# In that hash folder, there's many files with only numbers as names.


# Let's take one and see what's in it.
import _pickle as cPickle

infile = open("/Users/Alex/Code/hsmm-particlefilters/results/1834870540556565875/101")
thedict = cPickle.load(infile)
infile.close()
print(thedict.keys())

# Okay, all we care about is the mean track spit out by the particle filter.
# NOTE: there's a lag built-in, which is how we get better particle tracks.
# For this particular run, the lag is 15. "Fixed lag smoother"
meantrack = np.asarray(thedict['means'])

# The best tracks were run with RandomWalkFixedNoiseParallelSimplified
# using pose model PoseModel_5Joint_origweights_AW



# ==================================================
#                     PARALLEL
# ==================================================
#import experiments
#import os # HAAAAACK MATT GO LIKE DIS :()
# We first need to start some ipcluster instances (here, we do it locally)
#os.system("ipcluster start --n=4 &")
# Run a parallel fixed noise experiment
#experiments.RandomWalkFixedNoiseParallelSimplified((5,100)).run((5,100))



# ==================================================
#                THE EXPERIMENT CLASS
# ==================================================

# Let's talk about "experiments.py"
# Above, we just call some class from experiments, and hit run()
# To make a new experiment, you just need to subclass Experiment, and define a run() function.
# Let's break it down here.

class RandomWalkFixedNoiseFrozenTrack_AW_5Joints_simplified(Experiment):

	# Here's the run function.
    def run(self,frame_range):
    	# First things first, we have to figure out where our mouse image data is coming from
        datapath = os.path.join(root,"data/depth_videos/syllable_sorted-id-0 (usage)_original-id-42.mp4")

        # Then, we define how many particles we want to run
        # (particles are higher for the first step to start with a good guess)
        num_particles_firststep = 1024*10
        num_particles = 1024*4
        cutoff = 1024*4

        # This is the lag we saw above. It helps smooth tracks.
        lag = 15

        # This pose_model defines how many joints we are allowed to move.
        # The pose_model is allowed to restrict the set of joints that the particle
        # filter can move. Check out the methods and classes there.
        pose_model = pose_models.PoseModel_5Joint_origweights_AW()

        # The variances here define how widespread our guesses are, based off the previous guess.
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

        
        # This builds our OpenGL renderer
        # TODO: CHANGE THIS TO CUDA
        _build_mousescene(pose_model.scenefilepath)

        # Load in our real data, extracted from the Kinect
        images, xytheta = _load_data(datapath,frame_range)

        # This is a tiny tiny hack, where we use our knowledge of where we THINK
        # the mouse is to start off our particle filter. 
        pose_model.default_renderer_pose = \
            pose_model.default_renderer_pose._replace(theta_yaw=xytheta[0,2],x=xytheta[0,0],y=xytheta[0,1])
        pose_model.default_particle_pose = \
            pose_model.default_particle_pose._replace(theta_yaw=xytheta[0,2],x=xytheta[0,0],y=xytheta[0,1])

        # This is an interface between the OpenGL renderer and the particle filter,
        # that helps translate how good each guess is between the two classes.
        # Its main function is to make sure that we convert from "particle poses"
        # to "renderer poses". This is only important because we don't propose
        # over every degree of freedom in our model. 
        def log_likelihood(stepnum,im,poses):
            #return mp.get_likelihood()
            return ms.get_likelihood(im,particle_data=pose_model.expand_poses(poses),
                x=xytheta[stepnum,0],y=xytheta[stepnum,1],theta=xytheta[stepnum,2])/2000.


        # Okay, initialize the particle filter (we're using a random walk particle filter)
        pf = particle_filter.ParticleFilter(
                pose_model.particle_pose_tuple_len,
                cutoff,
                log_likelihood,
                [particle_filter.AR(
                    numlags=1,
                    previous_outputs=(pose_model.default_particle_pose,),
                    baseclass=lambda: pm.RandomWalk(noiseclass=lambda: pd.FixedNoise(randomwalk_noisechol))
                    ) for itr in range(num_particles_firststep)])

        # Do the first frame (we make a TON of guesses to get a good starting point)
        pf.step(images[0])

        # Now, we dial down the number of guesses, and the variance of the guesses
        # for subsequent frames. 
        pf.change_numparticles(num_particles)
        randomwalk_noisechol[:] = subsequent_randomwalk_noisechol[:]

        # Okay, now we start off the particle filter (which uses lags for smoothing)
        # by running #lags frames in to the future. This "prebuffers" our particle filter.
        for i in progprint_xrange(1,lag):
            pf.step(images[i])
        self.save_progress(pf,pose_model,datapath,frame_range,means=[])

        # Now, this is the bulk of the work. Run through all frames, saving the means
        # at every 5 frames
        means = []
        for i in progprint_xrange(lag,images.shape[0],perline=10):
            means.append(np.sum(pf.weights_norm[:,na] * np.array([p.track[i-lag] for p in pf.particles]),axis=0))
            print('\nsaved a mean for index %d with %d unique particles!\n' % \
                    (i-lag,len(np.unique([p.track[i-15][0] for p in pf.particles]))))

            pf.step(images[i])

            if (i % 5) == 0:
                self.save_progress(pf,pose_model,datapath,frame_range,means=means)

        # Save everything out once we're done. The means are the most important part right now!!
        self.save_progress(pf,pose_model,datapath,frame_range,means=means)




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