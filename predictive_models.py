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

    def copy(self):
        return copy.deepcopy(self)


###########################################
#  Wrappers for predictive_distributions  #
###########################################

class IID(PredictiveModel):
    def __init__(self,baseclass):
        self.sampler = baseclass()
        self.track = []

    def sample_next(self):
        self.track.append(self.sampler.sample_next())
        return self.track[-1]

    def __getattr__(self,name):
        return getattr(self.sampler,name)


class AR(IID):
    def __init__(self,numlags,baseclass):
        super(AR,self).__init__(baseclass)
        self.lagged_outputs = deque(maxlen=numlags)

    def sample_next(self):
        out = self.sampler.sample_next(lagged_outputs=self.lagged_outputs)
        self.lagged_outputs.appendleft(out)
        self.track.append(out)
        return out


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


def CRPSampler(PredictiveModel): # TODO
    pass


class _CRFIndexSampler(object):
    def __init__(self,alpha,gamma):
        self.table_samplers = defaultdict(lambda: _CRPIndexSampler(alpha))
        self.meta_table_sampler = _CRPIndexSampler(gamma)
        self.meta_table_assignments = defaultdict(lambda: defaultdict(self.meta_table_sampler.sample_next))

    def sample_next(self,restaurant_idx):
        return self.meta_table_assignments[restaurant_idx][self.table_samplers[restaurant_idx].sample_next()]


class HDPHMMSampler(PredictiveModel):
    def __init__(self,alpha,gamma,obs_sampler_factory):
        self.state_sampler = _CRFIndexSampler(alpha,gamma)
        self.dishes = defaultdict(obs_sampler_factory)
        self.stateseq = []
        self.track = []

    def sample_next(self,*args,**kwargs):
        cur_state = self.stateseq[-1] if len(self.stateseq) > 0 else 0
        self.stateseq.append(self.state_sampler.sample_next(cur_state))
        self.track.append(self.dishes[self.stateseq[-1]].sample_next(out=self.out,*args,**kwargs))
        return self.track[-1]


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
        self.track.append(self.dishes[self.stateseq[-1]].sample_next(*args,**kwargs))
        return self.track[-1]


### classes below are for ruling out self-transitions and probably need updating

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

