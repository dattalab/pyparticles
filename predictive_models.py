from __future__ import division
import numpy as np
from collections import defaultdict, deque
import abc, copy

from pymattutil.stats import sample_discrete

'''non-iid predictive samplers, mostly crp models'''
# PredictiveModels keep the 'track' and are used directly as particles
# in fact, they should just be called particles

class PredictiveModel(object):
    __metaclass__ = abc.ABCMeta

    # needs a self.track member

    @abc.abstractmethod
    def sample_next(self,*args,**kwargs):
        pass

    @abc.abstractmethod
    def copy(self):
        pass

#######################
#  Particle wrappers  #
#######################

class IID(PredictiveModel):
    def __init__(self,baseclass):
        self.sampler = baseclass()
        self.track = []

    def sample_next(self,*args,**kwargs):
        self.track.append(self.sampler.sample_next(*args,**kwargs))
        return self.track[-1]

    def __getattr__(self,name):
        return getattr(self.sampler,name)

    def __str__(self):
        return '%s(%s)' % (self.__class__.__name__,self.sampler.__str__())

    def copy(self):
        new = self.__new__(self.__class__)
        new.track = self.track[:]
        new.sampler = self.sampler.copy()
        return new


class AR(IID):
    def __init__(self,numlags,initial_obs,baseclass):
        assert len(initial_obs) == numlags
        super(AR,self).__init__(baseclass)
        self.lagged_outputs = deque(initial_obs,maxlen=numlags)

    def sample_next(self,*args,**kwargs):
        out = self.sampler.sample_next(lagged_outputs=self.lagged_outputs,*args,**kwargs)
        self.lagged_outputs.appendleft(out)
        self.track.append(out)
        return out

    def copy(self):
        new = super(AR,self).copy()
        new.lagged_outputs = self.lagged_outputs.__copy__()
        return new


##########
#  Meta  #
##########

class Mixture(PredictiveModel):
    def __init__(self,weights,components):
        self.weights = weights
        self.components = components

    def sample_next(self,*args,**kwargs):
        label = sample_discrete(self.weights)
        return self.components[label].sample_next(*args,**kwargs)

    def copy(self):
        new = self.__new__(self.__class__)
        new.weights = self.weights
        new.components = [c.copy() for c in self.components]
        return new

###################
#  'Dumb' models  #
###################

class RandomWalk(PredictiveModel):
    def __init__(self,noiseclass):
        self.noisesampler = noiseclass()

    def sample_next(self,lagged_outputs):
        y = lagged_outputs[0]
        return y + self.noisesampler.sample_next()

    def copy(self):
        new = self.__new__(self.__class__)
        new.noisesampler = self.noisesampler.copy()
        return new


class SideInfoRandomWalk(RandomWalk):
    def sample_next(self,sideinfo,lagged_outputs):
        y = lagged_outputs[0]
        y[:sideinfo.shape[0]] = sideinfo
        return y + self.noisesampler.sample_next()


class Momentum(PredictiveModel):
    def __init__(self,propmatrix,noiseclass):
        self.noisesampler = noiseclass()
        self.propmatrix = propmatrix # np.hstack((2*np.eye(ndim),-1*np.eye(ndim)))

    def sample_next(self,lagged_outputs):
        ys = np.concatenate(lagged_outputs)
        return self.propmatrix.dot(ys) + self.noisesampler.sample_next()

    def copy(self):
        new = self.__new__(self.__class__)
        new.noisesampler = self.noisesampler.copy()
        new.propmatrix = self.propmatrix
        return new


################
#  CRP models  #
################

class _CRPIndexSampler(object):
    def __init__(self,alpha):
        self.alpha = alpha
        self.assignments = []

    def sample_next(self):
        next_table = sample_discrete(self._get_distr())
        self.assignments.append(next_table)
        return next_table

    def _get_distr(self):
        return np.concatenate((np.bincount(self.assignments),(self.alpha,)))

    def copy(self):
        new = self.__new__(_CRPIndexSampler)
        new.alpha = self.alpha
        new.assignments = self.assignments[:]
        return new


def CRPSampler(PredictiveModel): # TODO
    pass


class _CRFIndexSampler(object):
    def __init__(self,alpha,gamma):
        self.table_samplers = defaultdict(lambda: _CRPIndexSampler(alpha))
        self.meta_table_sampler = _CRPIndexSampler(gamma)
        self.meta_table_assignments = defaultdict(lambda: defaultdict(self.meta_table_sampler.sample_next))

    def sample_next(self,restaurant_idx):
        return self.meta_table_assignments[restaurant_idx][self.table_samplers[restaurant_idx].sample_next()]

    def copy(self):
        new = self.__new__(_CRFIndexSampler)
        new.table_samplers = defaultdict(self.table_samplers.default_factory,
                ((s,t.copy()) for s,t in self.table_samplers.iteritems()))
        new.meta_table_sampler = self.meta_table_sampler.copy()
        new.meta_table_assignments = self.meta_table_assignments.copy()
        new.meta_table_assignments.default_factory = lambda: defaultdict(new.meta_table_sampler.sample_next)
        return new


class HDPHMMSampler(PredictiveModel):
    def __init__(self,alpha,gamma,obs_sampler_factory):
        self.state_sampler = _CRFIndexSampler(alpha,gamma)
        self.dishes = defaultdict(obs_sampler_factory)
        self.stateseq = []

    def sample_next(self,*args,**kwargs):
        cur_state = self.stateseq[-1] if len(self.stateseq) > 0 else 0
        self.stateseq.append(self.state_sampler.sample_next(cur_state))
        return self.dishes[self.stateseq[-1]].sample_next(*args,**kwargs)

    def copy(self):
        new = self.__new__(self.__class__)
        new.state_sampler = self.state_sampler.copy()
        new.dishes = defaultdict(self.dishes.default_factory,
                ((s,o.copy()) for s,o in self.dishes.iteritems()))
        new.stateseq = self.stateseq[:]
        return new

    def __str__(self):
        dishstr = '\n'.join('%d:\n%s\n' % (idx,self.dishes[idx])
                for idx in range(len(self.dishes)))
        return '%s(%s)\n%s\n' % (self.__class__.__name__,self.stateseq,dishstr)


class HDPHSMMSampler(HDPHMMSampler):
    def __init__(self,alpha,gamma,obs_sampler_factory,dur_sampler_factory):
        super(HDPHSMMSampler,self).__init__(alpha,gamma,obs_sampler_factory)
        self.dur_dishes = defaultdict(dur_sampler_factory)
        self.dur_counter = 0

    def sample_next(self,*args,**kwargs):
        if self.dur_counter > 0:
            self.stateseq.append(self.stateseq[-1])
            self.dur_counter -= 1
        else:
            cur_state = self.stateseq[-1] if len(self.stateseq) > 0 else 0
            self.stateseq.append(self.state_sampler.sample_next(cur_state))
            self.dur_counter = self.dur_dishes[self.stateseq[-1]].sample_next() - 1
        return self.dishes[self.stateseq[-1]].sample_next(*args,**kwargs)

    def copy(self):
        new = super(HDPHSMMSampler,self).copy()
        new.dur_dishes = defaultdict(self.dur_dishes.default_factory,
                ((s,d.copy()) for s,d in self.dur_dishes.iteritems()))
        new.dur_counter = self.dur_counter
        return new

### classes below are for ruling out self-transitions and NEED UPDATING

class _CRPIndexSamplerTaboo(_CRPIndexSampler):
    def __init__(self,alpha):
        self.alpha = alpha
        self.assignments = [0]

    def sample_next(self,taboo):
        next_table = sample_discrete(self._get_distr(taboo))
        self.assignments.append(next_table)
        return next_table

    def _get_distr(self,taboo):
        distn = super(_CRPIndexSamplerTaboo,self)._get_distr()
        distn[taboo] = 0
        return distn


class _CRFIndexSamplerNoSelf(_CRFIndexSampler):
    def __init__(self,alpha,gamma):
        self.table_samplers = defaultdict(lambda: _CRPIndexSampler(alpha))
        self.meta_table_sampler = _CRPIndexSamplerTaboo(gamma)
        self.meta_table_assignments = defaultdict(lambda: defaultdict(lambda: self.meta_table_sampler.sample_next))

    def sample_next(self,restaurant_idx):
        return self.meta_table_assignments[restaurant_idx]\
                [self.table_samplers[restaurant_idx].sample_next()](restaurant_idx)


class HDPHSMMNoSelfSampler(PredictiveModel):
    def __init__(self,alpha,gamma,obs_sampler_factory,dur_sampler_factory):
        self.state_sampler = _CRFIndexSamplerNoSelf(alpha,gamma)
        self.dishes = defaultdict(obs_sampler_factory)
        self.dur_dishes = defaultdict(dur_sampler_factory)
        self.stateseq = []
        self.dur_counter = 0

    def sample_next(self,*args,**kwargs):
        if self.dur_counter > 0:
            self.stateseq.append(self.stateseq[-1])
            self.dur_counter -= 1
        else:
            if len(self.stateseq) > 0:
                self.stateseq.append(self.state_sampler.sample_next(self.stateseq[-1]))
            else:
                self.stateseq.append(0)
            self.dur_counter = self.dur_dishes[self.stateseq[-1]].sample_next() - 1
        return self.dishes[self.stateseq[-1]].sample_next(*args,**kwargs)

