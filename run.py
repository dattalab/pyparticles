from __future__ import division
import numpy as np
na = np.newaxis
import os

from renderer.load_data import load_behavior_data
from renderer.renderer import MouseScene

import particle_filter as pf
from util.text import progprint_xrange, progprint

############################
#  computation parameters  #
############################

msNumRows, msNumCols = (32,32)
num_particles = msNumRows*msNumCols*5

####################
#  global objects  #
####################

# these global objects are shared for several purposes and convenient for
# interactive access. making them lazily loaded provides fast importing

ms = None # MouseScene object, global so that it's only built once
xytheta = images = None # offset sideinfo sequence and image sequence

def _build_mousescene(conf):
    scenefilepath = os.path.join(os.path.dirname(__file__),conf.pose_model.scenefilepath)

    global ms
    if ms is None:
        ms = MouseScene(scenefilepath, mouse_width=80, mouse_height=80, \
                        scale_width = 18.0, scale_height = 200.0,
                        scale_length = 18.0, \
                        numCols=msNumCols, numRows=msNumRows, useFramebuffer=False,showTiming=False)
        ms.gl_init()
    return ms

# TODO this should be in experiments
def _load_data_and_sideinfo(conf):
    datapath = os.path.join(os.path.dirname(__file__),conf.datapath)
    frame_range = conf.frame_range

    global xytheta, images
    if xytheta is None or images is None:
        xy = load_behavior_data(datapath,frame_range[1]+1,'centroid')[frame_range[0]:]
        theta = load_behavior_data(datapath,frame_range[1]+1,'angle')[frame_range[0]:]
        xytheta = np.concatenate((xy,theta[:,na]),axis=1)
        images = load_behavior_data(datapath,frame_range[1]+1,'images').astype('float32')[frame_range[0]:]
        images = np.array([image.T[::-1,:].astype('float32') for image in images])
    return xytheta, images

#############
#  running  #
#############

def run(conf,num_particles,cutoff,num_particles_firststep):
    # load 3D model object, data, sideinfo
    ms = _build_mousescene(conf)
    xytheta, images = _load_data_and_sideinfo(conf)

    # build the particle filter
    particlefilter = pf.ParticleFilter(
                        conf.pose_model.particle_pose_tuple_len,
                        cutoff,
                        conf.get_log_likelihood(ms,xytheta),
                        conf.get_initial_particles(num_particles_firststep))

    # first step is special
    particlefilter.step(images[0],sideinfo=xytheta[0])
    particlefilter.change_numparticles(num_particles)
    conf.first_step_done(particlefilter)

    # run the other steps
    for i in progprint_xrange(1,images.shape[0],perline=10):
        particlefilter.step(images[i],sideinfo=xytheta[i])

    return particlefilter

###########
#  utils  #
###########

def topk(items,scores,k):
    return [items[idx] for idx in np.argsort(scores)[:-(k+1):-1]]

def meantrack(particles,weights):
    track = np.array(particles[0].track)*weights[0,na]
    for p,w in zip(particles[1:],weights[1:]):
        track += np.array(p.track) * w
    return track

def render(conf,stepnum,particles):
    # might be slow; needs to do a full mousescene render pass
    from matplotlib import pyplot as plt
    _build_mousescene(conf), _load_data_and_sideinfo(conf)
    testimages = ms.get_likelihood(
        images[stepnum],
        particle_data=conf.pose_model.expand_poses(np.array([p.track[stepnum] for p in particles])),
        x=xytheta[stepnum,0],y=xytheta[stepnum,1],theta=xytheta[stepnum,2],
        return_posed_mice=True)[1]
    plt.imshow(np.hstack((images[stepnum],np.hstack(testimages))))
    plt.clim(0,300)

def movie(conf,track):#,outdir):
    # import Image
    # from util.general import scoreatpercentile
    _build_mousescene(conf), _load_data_and_sideinfo(conf)

    # _d = images.flatten()
    # scale = scoreatpercentile(_d[_d != 0],90,0)[0]

    rendered_images = ms.get_likelihood(
            np.zeros(images[0].shape),
            particle_data=conf.pose_model.expand_poses(track),
            x=xytheta[:,0],
            y=xytheta[:,1],
            theta=xytheta[:,2],
            return_posed_mice=True)[1]

    np.save('posed_mice.npy',rendered_images)

    # for i, (truth, rendered) in progprint(enumerate(zip(images,rendered_images)),total=len(rendered_images)):
        # Image.fromarray(np.clip(np.hstack((truth,rendered))*(255./scale),0,255.).astype('uint8'))\
        #         .save(os.path.join(outdir,'frame%d.png'%(i+conf.frame_range[0])))

##########
#  main  #
##########
if __name__ == '__main__':
    import experiment_configurations
    conf = experiment_configurations.Experiment3()
    _build_mousescene(conf), _load_data_and_sideinfo(conf)
    particlefilter = run(conf,10*1024,10*1024,30*1024)
