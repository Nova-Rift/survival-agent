"""Tiny shared transformer representation and MuZero-style latent lookahead.

Not full MuZero: exhaustive root actions plus top-2 prior expansion at depth two,
not MCTS. No pretrained language model, text reasoning or trained policy claimed.
"""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class HistoryAgent(nn.Module):
    def __init__(self, obs_width=5, history=4, width=32):
        super().__init__()
        if any(type(v) is not int for v in (obs_width,history,width)) or not (1<=obs_width<=33 and 1<=history<=32 and 4<=width<=256 and width%4==0):raise ValueError('Unsupported bounded history architecture')
        self.spec = {'obs_width':obs_width, 'history':history, 'width':width}
        self.embed = nn.Linear(5*obs_width*obs_width, width)
        self.positions = nn.Parameter(torch.zeros(1, history, width))
        layer = nn.TransformerEncoderLayer(width, 4, 2*width, dropout=0., batch_first=True)
        self.history_encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.transition = nn.Sequential(nn.Linear(width+5,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh())
        self.reward = nn.Linear(width,1)
        self.policy = nn.Linear(width,5)
        self.value = nn.Linear(width,1)

    def represent(self, history):
        x = self.embed(history.flatten(2)) + self.positions
        return torch.tanh(self.history_encoder(x)[:,-1])

    def predict(self, latent):
        return self.policy(latent), self.value(latent).squeeze(-1)

    def dynamics(self, latent, action):
        z = self.transition(torch.cat([latent,F.one_hot(action,5).to(latent.dtype)],dim=-1))
        return z, self.reward(z).squeeze(-1)

    @torch.no_grad()
    def plan(self, history, discount=.99):
        z = self.represent(history)
        batch, width = z.shape
        actions = torch.arange(5).repeat(batch)
        child, r1 = self.dynamics(z.repeat_interleave(5,dim=0),actions)
        policy, _ = self.predict(child)
        a2 = policy.topk(2,dim=-1).indices.flatten()
        leaf, r2 = self.dynamics(child.repeat_interleave(2,dim=0),a2)
        _, v2 = self.predict(leaf)
        q = r1.reshape(batch,5) + discount*(r2+discount*v2).reshape(batch,5,2).max(dim=-1).values
        return q.argmax(dim=-1), q

def history_start(obs, length=4):
    history = np.zeros((len(obs), length, *obs.shape[1:]),dtype=np.float32)
    history[:,-1] = obs
    return history

def history_append(history, observation):
    return np.concatenate([history[:,1:],observation[:,None]],axis=1)

def make_agent_population(template, count, *, weight_mode):
    """Explicit construction choice, not a population-learning implementation.
    Copies start with identical architecture/weights; neither mode shares history.
    The caller still owns each agent's inputs, RNG and future optimizer choices.
    """
    if type(count) is not int or count < 1:raise ValueError('Positive population required')
    if weight_mode=='shared':return [template]*count
    if weight_mode=='independent_copies':return [deepcopy(template) for _ in range(count)]
    raise ValueError('Choose shared or independent_copies explicitly')

def smoke_update(model, optimizer, batch):
    """One engineering update on real simulator transitions, NOT a training run.
    Policy targets imitate the local greedy baseline. Value targets are one-step
    bootstraps. This does not constitute a validated MuZero training recipe.
    """
    model.train()
    latent = model.represent(batch['history'])
    policy, value = model.predict(latent)
    next_latent, reward = model.dynamics(latent,batch['actions'])
    with torch.no_grad():
        next_repr = model.represent(batch['next_history'])
        _, next_value = model.predict(next_repr)
        value_target = batch['rewards'] + .99*batch['alive']*next_value
    losses = {
        'policy_imitation': F.cross_entropy(policy,batch['teacher_actions']),
        'value_bootstrap': F.mse_loss(value,value_target),
        'reward': F.mse_loss(reward,batch['rewards']),
        'latent_consistency': F.mse_loss(next_latent,next_repr),
    }
    loss = sum(losses.values())
    if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
    optimizer.zero_grad();loss.backward()
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():raise RuntimeError('Non-finite gradient')
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    optimizer.step()
    return {k:float(v.detach()) for k,v in losses.items()}

def save_checkpoint(*args, **kwargs):
    """Compatibility helper for combined environment/model state."""
    from survival_adapters.checkpoints import save_checkpoint as save
    return save(*args, **kwargs)


def load_checkpoint(*args, **kwargs):
    """Compatibility helper; model-only callers need not import an environment."""
    from survival_adapters.checkpoints import load_checkpoint as load
    return load(*args, **kwargs)
