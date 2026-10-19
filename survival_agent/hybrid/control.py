"""Minimal real-interaction actor/critic on frozen RSSM posterior features.

No imagination learner, value ensemble, planner or neuron recycling.
"""
from copy import deepcopy
import torch
from torch import nn
from .world_model import WorldModel


class PosteriorActorCritic(nn.Module):
    def __init__(self, world_model):
        super().__init__()
        if not isinstance(world_model, WorldModel) or world_model.spec.action.kind != 'discrete':
            raise ValueError('Discrete vector RSSM required')
        self.world_model = world_model.eval().requires_grad_(False)
        c = world_model.config
        width = c.deter+c.stoch*c.classes
        self.actor = nn.Linear(width, world_model.spec.action.count)
        self.value = nn.Linear(width, 1)
        self.target_value = deepcopy(self.value).eval().requires_grad_(False)

    def trainable_parameters(self):
        return list(self.actor.parameters())+list(self.value.parameters())

    def predict(self, features):
        if features.ndim != 2 or features.shape[1] != self.actor.in_features or not torch.isfinite(features).all():
            raise ValueError('Finite posterior feature batch required')
        # The policy optimizer must never alter the shared world representation.
        features = features.detach()
        return self.actor(features), self.value(features).squeeze(-1)

    @torch.no_grad()
    def update_target(self, ema=.99):
        if type(ema) not in (float, int) or not 0 <= ema < 1:
            raise ValueError('EMA must be in [0,1)')
        for target, value in zip(self.target_value.parameters(), self.value.parameters()):
            target.mul_(ema).add_(value, alpha=1-ema)
