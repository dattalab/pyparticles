from __future__ import division
import numpy as np
from collections import defaultdict, deque

from pymattutil.stats import sample_discrete

'''
CRP-based predictive samplers
'''

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

def CRPSampler(object):
    pass # TODO

class _CRFIndexSampler(object):
    def __init__(self,alpha,gamma):
        self.table_samplers = defaultdict(lambda: _CRPIndexSampler(alpha))
        self.meta_table_sampler = _CRPIndexSampler(gamma)
        self.meta_table_assignments = defaultdict(lambda: defaultdict(self.meta_table_sampler.sample_next))

    def sample_next(self,restaurant_idx):
        return self.meta_table_assignments[restaurant_idx][self.table_samplers[restaurant_idx].sample_next()]

class HDPHMMSampler(object):
    def __init__(self,alpha,gamma,obs_sampler_factory):
        self.state_sampler = _CRFIndexSampler(alpha,gamma)
        self.dishes = defaultdict(obs_sampler_factory)
        self.stateseq = []

    def sample_next(self,*args,**kwargs):
        cur_state = self.stateseq[-1] if len(self.stateseq) > 0 else 0
        self.stateseq.append(self.state_sampler.sample_next(cur_state))
        return self.dishes[self.stateseq[-1]].sample_next(out=self.out,*args,**kwargs)

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

class HDPHSMMARSampler(HDPHSMMSampler):
    def __init__(self,numlags,*args,**kwargs):
        super(HDPHSMMARSampler,self).__init__(*args,**kwargs)
        self.lagged_outputs = deque(maxlen=numlags)

    def sample_next(self):
        out = super(HDPHSMMARSampler,self).sample_next(lagged_outputs=self.lagged_outputs)
        self.lagged_outputs.appendleft(out)
        return out

# the next few classes are for ruling out self-transitions

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

class HDPHSMMNoSelfSampler(object):
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

