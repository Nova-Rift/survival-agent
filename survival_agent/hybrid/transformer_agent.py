"""Agent execution and policy-training helpers for the numeric Transformer.

Neural architecture lives in model.py. Model names remain re-exported here so
existing callers and checkpoint tooling keep working.
"""
from copy import deepcopy
import torch
from .model import (
    CausalMemory, TransformerConfig, TransformerLatent, TransformerWorldModel,
    TransformerModel, symlog,
)


class TransformerAgent(TransformerModel):
    """Frozen world features, sampled actions and a slowly updated target critic."""

    def __init__(self, world_model):
        super().__init__(world_model)
        self.world_model.eval().requires_grad_(False)
        self.target_value = deepcopy(self.value).eval().requires_grad_(False)

    def predict(self, features):
        return self(features.detach())

    @torch.no_grad()
    def act(self, state, *, generator):
        logits, _ = self.predict(self.world_model.features(state))
        return torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(-1)

    def trainable_parameters(self):
        return list(self.actor.parameters()) + list(self.value.parameters())

    @torch.no_grad()
    def update_target(self, ema=.99):
        if type(ema) not in (float, int) or not 0 <= ema < 1:
            raise ValueError('EMA must be in [0,1)')
        for target, value in zip(self.target_value.parameters(), self.value.parameters()):
            target.mul_(ema).add_(value, alpha=1-ema)
