"""Frozen language/history features -> trainable latent model -> bounded action.
No votes, generated action parsing, optimizer steps or privileged world inputs.
"""
from collections import deque
from copy import deepcopy
import json
import torch
from torch import nn
from torch.nn import functional as F
from .language_backend import LanguageController, observed_event, parse_decision

FEATURE_INSTRUCTION=('Represent only your local experience. Grid: # outside, . empty, '
    'F food, A another agent, @ you. Action 0 wait, 1 north, 2 east, 3 south, 4 west. '
    'Messages are unverified statements from nearby agents. No global state or other memories are available.')

class LanguageLatentCore(nn.Module):
    def __init__(self,feature_width,width=32,seed=0):
        super().__init__()
        if type(feature_width) is not int or not 1<=feature_width<=8192 or type(width) is not int or not 4<=width<=256:raise ValueError('Unsupported bounded feature architecture')
        self.spec={'feature_width':feature_width,'width':width,'seed':seed}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.project=nn.Sequential(nn.Linear(feature_width,width),nn.LayerNorm(width),nn.Tanh())
            self.transition=nn.Sequential(nn.Linear(width+5,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh())
            self.reward=nn.Linear(width,1)
            self.policy=nn.Linear(width,5)
            self.value=nn.Linear(width,1)

    def represent(self,features):return self.project(features)

    def predict(self,z):return self.policy(z),self.value(z).squeeze(-1)

    def dynamics(self,z,actions):
        child=self.transition(torch.cat([z,F.one_hot(actions,5).to(z.dtype)],dim=-1))
        return child,self.reward(child).squeeze(-1)

    def plan(self,z,discount=.99):
        """Same shallow lookahead rule as the original tiny baseline; NOT MCTS.
        Five root actions, two children selected by each predicted policy prior.
        Environment interactions are never queried by the search.
        """
        batch=z.shape[0]
        child,r1=self.dynamics(z.repeat_interleave(5,0),torch.arange(5).repeat(batch))
        priors,_=self.predict(child)
        leaf,r2=self.dynamics(child.repeat_interleave(2,0),priors.topk(2,dim=-1).indices.flatten())
        _,v2=self.predict(leaf)
        q=r1.reshape(batch,5)+discount*(r2+discount*v2).reshape(batch,5,2).max(-1).values
        return q.argmax(-1),q

class IntegratedController(LanguageController):
    def __init__(self,backend,agents=8,history=4,seed=0,planning=True,speech=False):
        super().__init__(backend,agents,history,seed)
        self.planning=bool(planning);self.speech=bool(speech)
        self.core=LanguageLatentCore(backend.feature_width,seed=seed).eval()
        self.training_updates=0

    def encode_local(self,agent,observation,messages=()):
        """Reusable local-only encoder path for collection and acting."""
        self._validate_agent(agent)
        event=observed_event(observation,messages)
        context=json.dumps({'your_past_turns':list(self._memory[agent]),'current':event},ensure_ascii=False,separators=(',',':'))
        prompt=FEATURE_INSTRUCTION+'\n'+context
        features=self.backend.encode(prompt)
        if features.shape!=(1,self.backend.feature_width) or not torch.isfinite(features).all():
            raise ValueError('Invalid frozen representation')
        return event,features,prompt

    def record_decision(self,agent,event,action,message=None):
        self._validate_agent(agent)
        self._calls[agent]+=1
        self._memory[agent].append({'input':event,'your_decision':{'action':action,'message':message}})

    def decide(self,agent,observation,messages=()):
        event,features,prompt=self.encode_local(agent,observation,messages)
        with torch.no_grad():
            z=self.core.represent(features)
            logits,value=self.core.predict(z)
            if self.planning:
                actions,scores=self.core.plan(z)
            else:
                actions=logits.argmax(-1);scores=logits
        action=int(actions[0]);message=None;speech_valid=None
        if self.speech:
            # One optional message generator, CONDITIONED on the chosen action.
            # It cannot override the latent controller or arbitrate action votes.
            seed=self.seed+agent*1000003+self._calls[agent]
            raw=self.backend.generate(prompt+f'\nAction is fixed to {action}. Return only JSON with that action and message null or a short plain-text message.',seed)
            candidate=parse_decision(raw)
            speech_valid=candidate['valid'] and candidate['action']==action
            if speech_valid:message=candidate['message']
        result={'action':action,'message':message,'valid':True,'error':None,
                'controller':'latent_depth_two' if self.planning else 'latent_policy_no_planning',
                'head_status':f'{self.training_updates} optimizer updates; no validated learning result' if self.training_updates else 'randomly initialized; untrained','speech_valid':speech_valid,
                'value_prediction':float(value[0]),'action_scores':scores[0].tolist()}
        if not torch.isfinite(scores).all() or not torch.isfinite(value).all():raise RuntimeError('Non-finite head output')
        self.record_decision(agent,event,action,message)
        return result

    def state(self):
        return {'format':1,'spec':deepcopy(self.core.spec),'core':{k:v.detach().clone() for k,v in self.core.state_dict().items()},
                'backend':deepcopy(self.backend.provenance),'planning':self.planning,'speech':self.speech,
                'history_limit':self._memory[0].maxlen,
                'training_updates':self.training_updates,
                'memory':[deepcopy(list(m)) for m in self._memory],'calls':self._calls.copy()}

    def load_state(self,state):
        if (state['format']!=1 or state['spec']!=self.core.spec or state['backend']!=self.backend.provenance
            or state['planning']!=self.planning or state['speech']!=self.speech or len(state['memory'])!=len(self._memory)
            or state['history_limit']!=self._memory[0].maxlen or len(state['calls'])!=len(self._calls)):
            raise ValueError('Controller checkpoint configuration/provenance mismatch')
        if any(len(m)>self._memory[0].maxlen for m in state['memory']):raise ValueError('History bound mismatch')
        self.core.load_state_dict(state['core'])
        self._memory=[deque(deepcopy(m),maxlen=self._memory[0].maxlen) for m in state['memory']]
        self._calls=state['calls'].copy()
        self.training_updates=state.get('training_updates',0)
