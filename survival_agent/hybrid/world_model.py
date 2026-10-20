"""Small categorical RSSM for real vector sequences, not a complete Dreamer agent.

Original PyTorch implementation; stabilization choices are documented in stage B.
Indices: posterior state at o[t+1] predicts r[t] and 1-terminated[t].
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .contracts import EnvSpec


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


@dataclass(frozen=True)
class ModelConfig:
    deter: int = 64
    hidden: int = 64
    stoch: int = 8
    classes: int = 8
    unimix: float = 0.01
    free_nats: float = 1.0
    dynamics_scale: float = 1.0
    representation_scale: float = 0.1

    def __post_init__(self):
        limits = {'deter': (8, 256), 'hidden': (8, 256), 'stoch': (1, 16), 'classes': (2, 32)}
        if any(type(getattr(self, k)) is not int or not lo <= getattr(self, k) <= hi
               for k, (lo, hi) in limits.items()):
            raise ValueError('Invalid bounded RSSM dimensions')
        for name, lo, hi in [('unimix', 0., .1), ('free_nats', 0., 20.),
                             ('dynamics_scale', 0., 10.), ('representation_scale', 0., 10.)]:
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError('Invalid bounded loss/distribution configuration')


@dataclass
class Latent:
    deter: torch.Tensor
    stoch: torch.Tensor

    def detach(self):
        return Latent(self.deter.detach(), self.stoch.detach())

    def clone(self):
        return Latent(self.deter.clone(), self.stoch.clone())


class WorldModel(nn.Module):
    def __init__(self, spec, config=ModelConfig()):
        super().__init__()
        if not isinstance(spec, EnvSpec) or not isinstance(config, ModelConfig):
            raise ValueError('Explicit environment/model specifications required')
        self.spec, self.config = spec, config
        self.action_width = spec.action.count if spec.action.kind == 'discrete' else len(spec.action.low)
        if self.action_width > 256:
            raise ValueError('Stage B bounds encoded action width to 256')
        c = config
        def network(inp, out):
            return nn.Sequential(nn.Linear(inp, c.hidden), nn.LayerNorm(c.hidden), nn.SiLU(), nn.Linear(c.hidden, out))
        self.encoder = network(spec.observation_width, c.hidden)
        self.core = nn.GRUCell(c.stoch*c.classes + self.action_width, c.deter)
        self.prior = network(c.deter, c.stoch*c.classes)
        self.posterior = network(c.deter+c.hidden, c.stoch*c.classes)
        feature = c.deter+c.stoch*c.classes
        self.decoder = network(feature, spec.observation_width)
        self.reward = network(feature, 1)
        self.continuation = network(feature, 1)

    def initial(self, batch_size):
        if type(batch_size) is not int or not 1 <= batch_size <= 32:
            raise ValueError('Batch size must be 1..32')
        c = self.config
        zeros = next(self.parameters()).new_zeros
        return Latent(zeros(batch_size, c.deter), zeros(batch_size, c.stoch, c.classes))

    def action_features(self, actions):
        if self.spec.action.kind == 'discrete':
            if actions.dtype != torch.int64 or actions.ndim != 1 or ((actions < 0) | (actions >= self.action_width)).any():
                raise ValueError('Exact int64 discrete action batch required')
            return F.one_hot(actions, self.action_width).to(dtype=next(self.parameters()).dtype)
        if actions.dtype != torch.float32 or actions.ndim != 2 or actions.shape[-1] != self.action_width or not torch.isfinite(actions).all() or (actions.abs() > 1).any():
            raise ValueError('Exact normalized float32 continuous action batch required')
        return actions

    def _check_state(self, state):
        c = self.config
        if not isinstance(state, Latent) or state.deter.ndim != 2:
            raise ValueError('Explicit recurrent/stochastic state required')
        batch = state.deter.shape[0]
        if not 1 <= batch <= 32 or state.deter.shape != (batch, c.deter) or state.stoch.shape != (batch, c.stoch, c.classes):
            raise ValueError('Invalid latent shape')
        reference = next(self.parameters())
        if any(x.dtype != reference.dtype or x.device != reference.device or not torch.isfinite(x).all()
               for x in [state.deter, state.stoch]):
            raise ValueError('Invalid latent dtype/device/value')
        return batch

    def probabilities(self, logits):
        p = F.softmax(logits, dim=-1)
        return (1-self.config.unimix)*p + self.config.unimix/self.config.classes

    def _latent(self, deter, logits, stochastic):
        p = self.probabilities(logits)
        if stochastic:
            index = torch.multinomial(p.reshape(-1, self.config.classes), 1).reshape(p.shape[:-1])
        else:
            index = p.argmax(-1)
        hard = F.one_hot(index, self.config.classes).to(p.dtype)
        # Straight-through categorical sample; mode is used only for diagnostics/tests.
        return Latent(deter, hard + (p - p.detach()))

    def prior_step(self, state, action, *, stochastic=True):
        batch = self._check_state(state)
        features = self.action_features(action)
        if features.shape[0] != batch:
            raise ValueError('Action/state batch mismatch')
        deter = self.core(torch.cat([state.stoch.flatten(1), features], -1), state.deter)
        logits = self.prior(deter).reshape(batch, self.config.stoch, self.config.classes)
        return self._latent(deter, logits, stochastic), logits

    def observe_step(self, state, previous_action, observation, is_first, *, stochastic=True):
        batch = self._check_state(state)
        if observation.shape != (batch, self.spec.observation_width) or observation.dtype != torch.float32 or not torch.isfinite(observation).all():
            raise ValueError('Exact finite float32 observation batch required')
        if is_first.dtype != torch.bool or is_first.shape != (batch,):
            raise ValueError('Exact per-stream reset flag required')
        # Reset both memory components AND previous-action features, independently per row.
        action = self.action_features(previous_action)
        if action.shape[0] != batch:
            raise ValueError('Action/state batch mismatch')
        keep = (~is_first).to(observation.dtype)
        deter = self.core(torch.cat([state.stoch.flatten(1)*keep[:, None], action*keep[:, None]], -1),
                          state.deter*keep[:, None])
        prior = self.prior(deter).reshape(batch, self.config.stoch, self.config.classes)
        encoded = self.encoder(symlog(observation))
        post = self.posterior(torch.cat([deter, encoded], -1)).reshape_as(prior)
        return self._latent(deter, post, stochastic), prior, post

    def features(self, state):
        return torch.cat([state.deter, state.stoch.flatten(1)], -1)

    def heads(self, state, *, decode=True):
        feature = self.features(state)
        result = {'reward_symlog': self.reward(feature).squeeze(-1),
                  'continuation_logit': self.continuation(feature).squeeze(-1)}
        if decode:
            result['observation_symlog'] = self.decoder(feature)
        return result

    def balanced_kl(self, posterior, prior):
        q, p = self.probabilities(posterior), self.probabilities(prior)
        # Aggregate category KL across stochastic variables BEFORE applying free nats.
        tiny = torch.finfo(q.dtype).tiny
        def kl(a, b):
            return (a*(a.clamp_min(tiny).log()-b.clamp_min(tiny).log())).sum((-1, -2))
        dyn = kl(q.detach(), p).clamp_min(self.config.free_nats)
        rep = kl(q, p.detach()).clamp_min(self.config.free_nats)
        return dyn, rep


# Shared trainers now live with the full hybrid in model.py. The historical GRU
# architecture above is preserved; existing imports continue to use these names.
from .model import (sequence_batch, sequence_loss, WorldModelTrainer,
                    prediction_diagnostics)
