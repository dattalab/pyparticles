from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
plt.ion()
import cPickle

import pyhsmm
from pyhsmm.util.text import progprint_xrange

import pyhsmm.plugins.autoregressive.models as m
import pyhsmm.plugins.autoregressive.distributions as d

# TODO labels

##############
#  get data  #
##############
with open('ALL_RESULTS','r') as infile:
   dct = cPickle.load(infile)

dct = dict((k,np.array(v)) for k,v in dct.iteritems())
tracks = []
tracks.append(dct['5.1000.UPTO_1000'][:1000])
tracks.append(dct['5870.6200'])
tracks.append(dct['6300.7600.UPTO_1022'][:1022])
tracks.append(dct['7750.8500.UP_TO_483_AND_AFTER_578'][:483])
tracks.append(dct['7750.8500.UP_TO_483_AND_AFTER_578'][579:])
tracks.append(dct['1550.2200.UP_TO_355'][:355])
tracks.append(dct['5000.5400'])

# cut out scales
# pose spec is
# ['x','y','theta_yaw',
#     'z','theta_roll','s_w','s_l','s_h',
#     'psi_z3','psi_y4','psi_z4','psi_y5','psi_z5'])
# tracks = [np.concatenate((track[:,:3],track[:,-5:]),axis=1) for track in tracks]

# only keep angles
tracks = [track[:,-5:] for track in tracks]


##############
#  parallel  #
##############
import pyhsmm.parallel as parallel
parallel.alldata = dict((k,v) for k,v in enumerate(tracks))

from IPython.parallel import Client
dv = Client()[:]
dv['alldata'] = parallel.alldata

# ###################################
# #  get empirical hyperparameters  #
# ###################################
dtracks = [np.diff(t,axis=0) for t in tracks]
dmu = (np.array([dt.mean(0) for dt in dtracks])*np.array([len(dt) for dt in dtracks])[:,None]).sum(0)/sum(map(len,dtracks))
dcov = np.zeros((dmu.shape[0],dmu.shape[0]))
for dt in dtracks:
   for v in dt:
       dcov += np.outer(v,v)
dcov /= sum(map(len,dtracks))
dcov -= np.outer(dmu,dmu)

# ###################
# #  build a model  #
# ###################

Nmax = 50
model = m.ARHSMM(
       nlags=2,
       alpha_a_0=0.5,alpha_b_0=3.,
       gamma_a_0=0.5,gamma_b_0=3.,
       init_state_concentration=6.,
       obs_distns=[d.MNIW(len(dmu)+2,5*np.diag(np.diag(dcov)),np.zeros((len(dmu),2*len(dmu))),10*np.eye(2*len(dmu)))
                                   for state in range(Nmax)],
       # dur_distns=[pyhsmm.basic.distributions.GeometricDuration(100,2000) for state in range(Nmax)],
       # dur_distns=[pyhsmm.basic.distributions.PoissonDuration(3*20,3) for state in range(Nmax)],
       dur_distns=[pyhsmm.basic.distributions.NegativeBinomialDuration(10*10.,1./10.,3*10.,1*10.)
                           for state in range(Nmax)],
       )

for i,t in enumerate(tracks):
   model.add_data_parallel(i)

########################
#  try some inference  #
########################
plt.figure()
for itr in progprint_xrange(200):
   if itr % 5 == 0:
       plt.gcf().clf()
       model.plot()
       plt.ion()
       plt.draw()
       plt.ioff()
   model.resample_model_parallel()
