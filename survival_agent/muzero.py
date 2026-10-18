"""Distinct minimal MuZero-style PUCT/search-target/replay/unroll option.
Not the actor-critic baseline and not a reproduction of full-scale MuZero.
"""
from dataclasses import dataclass,field
from collections import deque
from copy import deepcopy
import hashlib
import math
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .agent import HistoryAgent
from .integrated import LanguageLatentCore

class MuZeroModel(nn.Module):
    def __init__(self,encoder_mode='scratch_history',feature_width=32,seed=0,encoder_provenance=None,language_encoder_training='frozen'):
        super().__init__()
        if type(feature_width) is not int or not 1<=feature_width<=8192:raise ValueError('Unsupported bounded feature width')
        if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Invalid seed')
        if language_encoder_training!='frozen':raise NotImplementedError('Pretrained language fine-tuning is not implemented; never silently frozen')
        if encoder_mode not in ['scratch_history','frozen_language_features']:raise ValueError('Unknown encoder mode')
        if encoder_mode=='frozen_language_features' and not encoder_provenance:raise ValueError('Explicit frozen encoder provenance required')
        self.spec={'encoder_mode':encoder_mode,'feature_width':feature_width,'seed':seed,
                   'encoder_provenance':deepcopy(encoder_provenance),'language_encoder_training':language_encoder_training}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.core=HistoryAgent() if encoder_mode=='scratch_history' else LanguageLatentCore(feature_width,seed=seed)
            self.terminal=nn.Linear(32,1)

    def initial(self,inputs):
        latent=self.core.represent(inputs)
        policy,value=self.core.predict(latent)
        return latent,policy,value

    def recurrent(self,latent,actions):
        latent,reward=self.core.dynamics(latent,actions)
        policy,value=self.core.predict(latent)
        return latent,reward,policy,value,self.terminal(latent).squeeze(-1)

@dataclass
class Node:
    prior: float = 1.
    latent: object = None
    reward: float = 0.
    terminal: bool = False
    prediction: float = 0.
    visits: int = 0
    value_sum: float = 0.
    children: dict = field(default_factory=dict)

    @property
    def value(self):return self.value_sum/self.visits if self.visits else 0.

def expand(node,logits):
    probabilities=logits[0].softmax(-1).detach().numpy()
    if not np.isfinite(probabilities).all():raise RuntimeError('Non-finite search prior')
    node.children={a:Node(prior=float(p)) for a,p in enumerate(probabilities)}

@torch.no_grad()
def mcts(model,inputs,simulations=8,max_depth=4,gamma=.99,c_puct=1.25,root_terminal=False,terminal_threshold=.9):
    """Single-agent survival-value backup: no adversarial sign flips. Search has
    NO world/simulator handle. Hypothetical termination comes from a learned head.
    Actual root death can be supplied from that agent's own alive observation.
    """
    if not 1<=simulations<=64 or not 1<=max_depth<=16:raise ValueError('Bounded search budget required')
    if not 0<gamma<=1 or c_puct<=0 or not 0<terminal_threshold<1:raise ValueError('Invalid search settings')
    if root_terminal:return {'visits':np.zeros(5,dtype=np.int64),'policy':np.zeros(5,dtype=np.float32),'value':0.,'root':Node(terminal=True)}
    latent,policy,value=model.initial(inputs)
    root=Node(latent=latent,prediction=float(value[0]));expand(root,policy)
    for _ in range(simulations):
        node=root;path=[root];depth=0
        while node.children and not node.terminal and depth<max_depth:
            def score(action):
                child=node.children[action]
                q=child.reward+(0 if child.terminal else gamma*child.value)
                return q+c_puct*child.prior*math.sqrt(node.visits+1)/(1+child.visits)
            action=max(node.children,key=lambda a:(score(a),-a))
            child=node.children[action];path.append(child);depth+=1
            if child.latent is None:
                z,reward,logits,value,terminal=model.recurrent(node.latent,torch.tensor([action]))
                child.latent=z;child.reward=float(reward[0]);child.prediction=float(value[0])
                child.terminal=float(terminal[0].sigmoid())>=terminal_threshold
                if not math.isfinite(child.reward+child.prediction):raise RuntimeError('Non-finite search prediction')
                if not child.terminal:expand(child,logits)
                node=child;break
            node=child
        backed=0. if node.terminal else node.prediction
        for item in reversed(path):
            item.value_sum+=backed;item.visits+=1
            backed=item.reward+(0 if item.terminal else gamma*backed)
    counts=np.array([root.children[a].visits for a in range(5)],dtype=np.int64)
    if counts.sum()!=simulations:raise RuntimeError('Search visit accounting failed')
    return {'visits':counts,'policy':(counts/counts.sum()).astype(np.float32),'value':root.value,'root':root}

@dataclass
class Episode:
    inputs: np.ndarray # T+1 own histories or frozen local-language features.
    actions: np.ndarray
    rewards: np.ndarray
    policies: np.ndarray # Actual normalized MCTS visit counts at T roots.
    root_values: np.ndarray # T+1, final truncation bootstrap or terminal zero.
    terminated: np.ndarray
    truncated: np.ndarray

    def validate(self):
        arrays=[self.inputs,self.actions,self.rewards,self.policies,self.root_values,self.terminated,self.truncated]
        if any(not isinstance(x,np.ndarray) for x in arrays):raise ValueError('Episode arrays required')
        if self.actions.ndim!=1 or not 1<=self.actions.size<=4096:raise ValueError('Bounded one-dimensional actions required')
        n=len(self.actions)
        if self.actions.dtype.kind not in 'iu' or self.terminated.dtype!=np.bool_ or self.truncated.dtype!=np.bool_:raise ValueError('Invalid episode dtype')
        if any(x.shape!=(n,) for x in [self.rewards,self.terminated,self.truncated]) or self.root_values.shape!=(n+1,):raise ValueError('Exact transition shapes required')
        if self.inputs.ndim not in (2,5) or self.inputs.shape[0]!=n+1 or self.inputs.nbytes>16*1024*1024:raise ValueError('Invalid bounded encoder inputs')
        if self.inputs.ndim==2 and not 1<=self.inputs.shape[1]<=8192:raise ValueError('Invalid feature shape')
        if self.inputs.ndim==5 and self.inputs.shape[1:]!=(4,5,5,5):raise ValueError('Invalid history shape')
        if any(x.dtype not in (np.dtype('float32'),np.dtype('float64')) for x in [self.inputs,self.rewards,self.policies,self.root_values]):raise ValueError('Floating episode arrays required')
        if n<1 or len(self.inputs)!=n+1 or self.policies.shape!=(n,5) or len(self.root_values)!=n+1:raise ValueError('Episode length mismatch')
        if any(len(x)!=n for x in [self.rewards,self.terminated,self.truncated]):raise ValueError('Transition length mismatch')
        if (self.terminated&self.truncated).any() or (self.terminated[:-1]|self.truncated[:-1]).any() or not (self.terminated[-1] or self.truncated[-1]):raise ValueError('One episode boundary at end required')
        if not np.allclose(self.policies.sum(-1),1) or (self.policies<0).any():raise ValueError('Visit targets must normalize')
        if ((self.actions<0)|(self.actions>=5)).any():raise ValueError('Invalid action')
        if not all(np.isfinite(x).all() for x in [self.inputs,self.rewards,self.policies,self.root_values]):raise ValueError('Non-finite episode')

def value_target(episode,start,td_steps,gamma):
    n=len(episode.actions)
    if start>=n:return 0. if episode.terminated[-1] else float(episode.root_values[n])
    value=0.
    for j in range(start,min(start+td_steps,n)):
        value+=gamma**(j-start)*float(episode.rewards[j])
        if episode.terminated[j]:return value
        if episode.truncated[j]:return value+gamma**(j-start+1)*float(episode.root_values[j+1])
    end=min(start+td_steps,n)
    return value+gamma**(end-start)*float(episode.root_values[end])

class EpisodeReplay:
    def __init__(self,seed=0,capacity=128):
        if capacity<1:raise ValueError('Positive replay capacity required')
        self.episodes=[];self.rng=np.random.default_rng(seed);self.capacity=capacity

    def add(self,episode):
        if self.episodes and episode.inputs.shape[1:]!=self.episodes[0].inputs.shape[1:]:raise ValueError('Cannot mix encoder input spaces')
        episode.validate();self.episodes.append(deepcopy(episode))
        self.episodes=self.episodes[-self.capacity:]

    def sample(self,batch_size=2,unroll=2,td_steps=3,gamma=.99):
        if not self.episodes or not 1<=batch_size<=32 or not 1<=unroll<=8 or td_steps<1:raise ValueError('Invalid bounded replay sample')
        rows=[]
        for _ in range(batch_size):
            ep=self.episodes[int(self.rng.integers(len(self.episodes)))];start=int(self.rng.integers(len(ep.actions)))
            row={'inputs':ep.inputs[start],'actions':np.zeros(unroll,np.int64),'reward':np.zeros(unroll,np.float32),
                 'terminal':np.zeros(unroll,np.float32),'transition_mask':np.zeros(unroll,np.float32),
                 'value':np.zeros(unroll+1,np.float32),'value_mask':np.zeros(unroll+1,np.float32),
                 'policy':np.zeros((unroll+1,5),np.float32),'policy_mask':np.zeros(unroll+1,np.float32)}
            for k in range(unroll+1):
                index=start+k;n=len(ep.actions)
                if index<n:
                    row['policy'][k]=ep.policies[index];row['policy_mask'][k]=1
                    row['value'][k]=value_target(ep,index,td_steps,gamma);row['value_mask'][k]=1
                elif index==n and ep.terminated[-1]:row['value_mask'][k]=1 # Known terminal value zero.
                if k and index-1<n:
                    j=index-1;row['actions'][k-1]=ep.actions[j];row['reward'][k-1]=ep.rewards[j]
                    row['terminal'][k-1]=ep.terminated[j];row['transition_mask'][k-1]=1
            rows.append(row)
        return {key:torch.from_numpy(np.stack([row[key] for row in rows])) for key in rows[0]}

def masked_mean(values,mask):return (values*mask).sum()/mask.sum().clamp_min(1)

def unroll_loss(model,batch):
    latent,policy,value=model.initial(batch['inputs'])
    policies=[policy];values=[value];rewards=[];terminals=[]
    for k in range(batch['actions'].shape[1]):
        latent,reward,policy,value,terminal=model.recurrent(latent,batch['actions'][:,k])
        policies.append(policy);values.append(value);rewards.append(reward);terminals.append(terminal)
    policy=torch.stack(policies,1);value=torch.stack(values,1)
    reward=torch.stack(rewards,1);terminal=torch.stack(terminals,1)
    losses={'policy_visits':masked_mean(-(batch['policy']*policy.log_softmax(-1)).sum(-1),batch['policy_mask']),
            'value':masked_mean((value-batch['value']).square(),batch['value_mask']),
            'reward':masked_mean((reward-batch['reward']).square(),batch['transition_mask']),
            'termination':masked_mean(F.binary_cross_entropy_with_logits(terminal,batch['terminal'],reduction='none'),batch['transition_mask'])}
    return losses['policy_visits']+.25*losses['value']+losses['reward']+.1*losses['termination'],losses

class MuZeroLearner:
    def __init__(self,model,seed=0):
        self.model=model;self.replay=EpisodeReplay(seed);self.optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        self.updates=0;self.episodes_collected=0

    def update(self,batch_size=2,unroll=2,td_steps=3):
        self.model.train();batch=self.replay.sample(batch_size,unroll,td_steps)
        self.optimizer.zero_grad();loss,parts=unroll_loss(self.model,batch)
        if not torch.isfinite(loss):raise RuntimeError('Non-finite MuZero-style loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):raise RuntimeError('Non-finite gradient')
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.)
        self.optimizer.step();self.updates+=1;self.model.eval()
        return {k:float(v.detach()) for k,v in parts.items()}|{'total':float(loss.detach()),'gradient_norm':float(norm)}

    def state(self):
        hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['muzero.py','agent.py','integrated.py']}
        return deepcopy({'format':1,'algorithm':'minimal_muzero_style_puct_visits_replay_unroll','spec':self.model.spec,
            'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),'replay':self.replay.episodes,
            'replay_rng':self.replay.rng.bit_generator.state,'capacity':self.replay.capacity,
            'torch_rng':torch.get_rng_state(),'updates':self.updates,'episodes_collected':self.episodes_collected,
            'source_hashes':hashes,'torch_version':str(torch.__version__)})

    def load_state(self,state):
        current=self.state()
        if state['format']!=1 or state['spec']!=self.model.spec or state['source_hashes']!=current['source_hashes'] or state['torch_version']!=current['torch_version']:raise ValueError('MuZero checkpoint provenance mismatch')
        self.model.load_state_dict(state['model']);self.optimizer.load_state_dict(state['optimizer'])
        self.replay.episodes=deepcopy(state['replay']);self.replay.capacity=state['capacity']
        self.replay.rng.bit_generator.state=deepcopy(state['replay_rng']);torch.set_rng_state(state['torch_rng'])
        self.updates=state['updates'];self.episodes_collected=state['episodes_collected']

def collect_episode(*args, **kwargs):
    """Compatibility entry point for the survival-specific collector adapter."""
    from survival_adapters.collectors import collect_episode as collect
    return collect(*args, **kwargs)
