"""Replay-seeded Dreamer-style imagination and real-replay continuous TD learning.

Original reduced PyTorch implementation. The pinned Dreamer score-function
objective is used for both action types, with detached advantages/features.
World-model parameters receive no gradients from these policy or Q updates.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
import numpy as np
import torch
from torch import nn
from .policy import sample_action, log_probability
from .world_model import sequence_batch, symexp, WorldModelTrainer


@dataclass(frozen=True)
class LearningConfig:
    horizon: int = 5
    gamma: float = .99
    lambda_: float = .95
    entropy: float = 3e-4
    learning_rate: float = 3e-4
    target_ema: float = .99
    gradient_clip: float = 10.
    slow_value_regularizer: float = 1.
    distillation_weight: float = 0.
    max_target_age: int = 10

    def __post_init__(self):
        if type(self.horizon) is not int or not 1 <= self.horizon <= 32:
            raise ValueError('Invalid imagination horizon')
        if type(self.max_target_age) is not int or not 0 <= self.max_target_age <= 10000:
            raise ValueError('Invalid planner target age')
        for key in ('gamma','lambda_','entropy','learning_rate','target_ema','gradient_clip',
                    'slow_value_regularizer','distillation_weight'):
            if not math.isfinite(getattr(self,key)):
                raise ValueError('Nonfinite learning configuration')
        if not (0 <= self.gamma <= 1 and 0 <= self.lambda_ <= 1 and 0 <= self.entropy <= 1
                and 0 < self.learning_rate <= .01 and 0 <= self.target_ema < 1
                and 0 < self.gradient_clip <= 100 and 0 <= self.slow_value_regularizer <= 10
                and 0 <= self.distillation_weight <= 10):
            raise ValueError('Invalid learning configuration')


def lambda_returns(rewards, continuations, values, gamma, lambda_):
    """B x H transitions, B x (H+1) values; gamma is applied exactly once."""
    if rewards.ndim != 2 or continuations.shape != rewards.shape or values.shape != (rewards.shape[0],rewards.shape[1]+1):
        raise ValueError('Invalid lambda-return shapes')
    if not 0 <= gamma <= 1 or not 0 <= lambda_ <= 1 or not all(torch.isfinite(x).all() for x in (rewards,continuations,values)) or ((continuations<0)|(continuations>1)).any():
        raise ValueError('Invalid lambda-return values')
    result,tail = [],values[:,-1]
    for t in reversed(range(rewards.shape[1])):
        tail = rewards[:,t]+gamma*continuations[:,t]*((1-lambda_)*values[:,t+1]+lambda_*tail)
        result.append(tail)
    return torch.stack(result[::-1],1)


@torch.no_grad()
def replay_states(model, rows):
    """Reconstruct last real transition from full episode prefixes, never reset obs.

    One start per row bounds imagination batch size. Rows originate in real replay;
    caller-defined arbitrary latent vectors are not accepted by the learner.
    """
    world = model.world_model
    batch = sequence_batch(rows,world.spec)
    device = next(model.parameters()).device
    batch = {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
    actions,obs = batch['actions'],batch['observations']
    n,length = actions.shape[:2]
    state,_,_ = world.observe_step(world.initial(n),torch.zeros_like(actions[:,0]),obs[:,0],
                                   torch.ones(n,dtype=torch.bool,device=device),stochastic=False)
    previous = state
    for t in range(length):
        previous = state
        state,_,_ = world.observe_step(state,actions[:,t],obs[:,t+1],
                                      torch.zeros(n,dtype=torch.bool,device=device),stochastic=False)
    return previous.detach(),state.detach(),batch


def replay_reference(rows):
    digest = hashlib.sha256()
    for row in rows:
        for key in ('observations','actions','terminated','truncated','masks'):
            if row.get(key) is not None:
                value = row[key]
                digest.update(key.encode()+str(value.shape).encode()+str(value.dtype).encode()+value.tobytes())
    return digest.hexdigest()


def planner_target(rows, plan, *, version, step, settings):
    """Detached targets retain their real-history identity and planner metadata."""
    return {'history_sha256':replay_reference(rows),'target':plan.target.detach().clone(),
            'version':version,'step':step,'settings':deepcopy(settings),
            'planner':plan.diagnostics['planner']}


class HybridLearner:
    def __init__(self, model, config=LearningConfig()):
        self.model,self.config = model,config
        self.world_trainer = WorldModelTrainer(model.world_model,learning_rate=config.learning_rate)
        self.target_value = deepcopy(model.value).eval().requires_grad_(False)
        self.target_q = deepcopy(model.q_ensemble).eval().requires_grad_(False)
        self.policy_parameters = list(model.actor.parameters())+list(model.value.parameters())
        self.policy_optimizer = torch.optim.Adam(self.policy_parameters,lr=config.learning_rate)
        self.q_optimizer = torch.optim.Adam(model.q_ensemble.parameters(),lr=config.learning_rate) if len(model.q_ensemble) else None
        self.return_low,self.return_high = 0.,1.
        self.updates,self.model_version = 0,0

    def world_step(self, rows):
        result = self.world_trainer.step(rows)
        self.model_version += 1
        return result

    def _step(self, loss, optimizer, parameters):
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite hybrid loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(parameters,self.config.gradient_clip,error_if_nonfinite=True)
        optimizer.step()

    @torch.no_grad()
    def _targets(self):
        for online,target in ((self.model.value,self.target_value),(self.model.q_ensemble,self.target_q)):
            for source,destination in zip(online.parameters(),target.parameters()):
                destination.lerp_(source,1-self.config.target_ema)

    def imagination_step(self, rows, target=None):
        c,model = self.config,self.model
        model.world_model.zero_grad(set_to_none=True)
        state,_,_ = replay_states(model,rows)
        features,actions,rewards,continuations = [],[],[],[]
        with torch.no_grad():
            for _ in range(c.horizon):
                features.append(model.world_model.features(state))
                action = sample_action(model,features[-1])
                actions.append(action)
                state,_ = model.world_model.prior_step(state,action,stochastic=True)
                heads = model.world_model.heads(state,decode=False)
                rewards.append(symexp(heads['reward_symlog']))
                continuations.append(heads['continuation_logit'].sigmoid())
            features.append(model.world_model.features(state))
            feature = torch.stack(features,1)
            slow_values = self.target_value(feature).squeeze(-1)
            rewards,continuations = torch.stack(rewards,1),torch.stack(continuations,1)
            returns = lambda_returns(rewards,continuations,slow_values,c.gamma,c.lambda_)
            low,high = torch.quantile(returns.flatten(),torch.tensor([.05,.95],device=returns.device))
            self.return_low = .99*self.return_low+.01*float(low)
            self.return_high = .99*self.return_high+.01*float(high)
            advantage = (returns-slow_values[:,:-1])/max(1.,self.return_high-self.return_low)
            weights = torch.cat([torch.ones_like(continuations[:,:1]),
                                 (c.gamma*continuations[:,:-1]).cumprod(1)],1)
        logp,entropy = log_probability(model,feature[:,:-1],torch.stack(actions,1))
        actor_loss = -(weights*(logp*advantage+ c.entropy*entropy)).mean()
        values = model.value(feature[:,:-1]).squeeze(-1)
        value_loss = (weights*((values-returns).square()+c.slow_value_regularizer*(values-slow_values[:,:-1]).square())).mean()
        distill = values.new_zeros(())
        if c.distillation_weight and target is not None:
            if (target['history_sha256'] != replay_reference(rows) or target['version'] != self.model_version
                    or not 0 <= self.updates-target['step'] <= c.max_target_age):
                raise ValueError('Stale or mismatched planner target')
            teacher = target['target'].detach().to(values.device)
            if not torch.isfinite(teacher).all():
                raise ValueError('Nonfinite planner target')
            policy = model.policy(feature[:,0])
            if model.spec.action.kind == 'discrete':
                if target['planner'] != 'mcts' or teacher.shape != policy['logits'].shape or (teacher < 0).any() or not torch.allclose(teacher.sum(-1),torch.ones_like(teacher[:,0])):
                    raise ValueError('Invalid visit target')
                logits = policy['logits']
                if any(row.get('masks') is not None for row in rows):
                    masks = torch.as_tensor(np.stack([row['masks'][-2] for row in rows]),device=values.device)
                    if (teacher.masked_select(~masks) != 0).any():
                        raise ValueError('Illegal target probability')
                    logits = logits.masked_fill(~masks,-1e9)
                distill = -(teacher*logits.log_softmax(-1)).sum(-1).mean()
            else:
                if target['planner'] != 'mpc' or teacher.shape != policy['mean'].shape or (teacher.abs()>1).any():
                    raise ValueError('Invalid continuous action target')
                distill = (policy['mean'].tanh()-teacher).square().mean()
        loss = actor_loss+value_loss+c.distillation_weight*distill
        self._step(loss,self.policy_optimizer,self.policy_parameters)
        self._targets()
        self.updates += 1
        return {'actor_loss':float(actor_loss.detach()),'value_loss':float(value_loss.detach()),
                'distillation_loss':float(distill.detach()),'imagined_return':float(returns.mean())}

    def td_step(self, rows):
        if self.q_optimizer is None:
            raise ValueError('Continuous TD requires Q heads')
        model,c = self.model,self.config
        model.world_model.zero_grad(set_to_none=True)
        current,next_state,batch = replay_states(model,rows)
        features = model.world_model.features(current)
        with torch.no_grad():
            next_features = model.world_model.features(next_state)
            action = sample_action(model,next_features)
            x = torch.cat([next_features,action],-1)
            # TD-MPC2 reference: minimum of two uniformly subsampled target Qs.
            pair = torch.randperm(len(self.target_q),device=x.device)[:2]
            q = torch.stack([head(x).squeeze(-1) for head in self.target_q])[pair].min(0).values
            target = batch['rewards'][:,-1]+c.gamma*batch['continuation'][:,-1]*q
        prediction = model.q_values(features,batch['actions'][:,-1])
        loss = (prediction-target[None]).square().mean()
        self._step(loss,self.q_optimizer,list(model.q_ensemble.parameters()))
        self._targets()
        self.updates += 1
        return {'q_loss':float(loss.detach()),'td_target':target.detach().cpu().tolist()}

    def state_dict(self):
        return {'model':self.model.state_dict(),'target_value':self.target_value.state_dict(),
                'target_q':self.target_q.state_dict(),'world_optimizer':self.world_trainer.optimizer.state_dict(),
                'policy_optimizer':self.policy_optimizer.state_dict(),
                'q_optimizer':None if self.q_optimizer is None else self.q_optimizer.state_dict(),
                'return_low':self.return_low,'return_high':self.return_high,'updates':self.updates,
                'model_version':self.model_version,'world_updates':self.world_trainer.updates}

    def load_state_dict(self, state):
        self.model.load_state_dict(state['model'],strict=True)
        self.target_value.load_state_dict(state['target_value'],strict=True)
        self.target_q.load_state_dict(state['target_q'],strict=True)
        self.world_trainer.optimizer.load_state_dict(state['world_optimizer'])
        self.policy_optimizer.load_state_dict(state['policy_optimizer'])
        if self.q_optimizer is not None:
            self.q_optimizer.load_state_dict(state['q_optimizer'])
        for key in ('return_low','return_high','updates','model_version'):
            setattr(self,key,state[key])
        self.world_trainer.updates = state['world_updates']
