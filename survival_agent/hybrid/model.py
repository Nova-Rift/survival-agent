"""Complete numeric Transformer architecture, independent of agent execution.

Defines configuration, explicit latent tensors, causal memory, world-model
predictions and actor/value outputs. Callers supply history and receive tensors;
this module does not select environment actions, freeze weights, update target
networks, collect experience or run an optimizer. Attention uses PyTorch layers.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .contracts import EnvSpec


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


@dataclass(frozen=True)
class TransformerConfig:
    deter: int = 32             # Transformer embedding width.
    hidden: int = 32            # Width inside observation/prediction heads.
    stoch: int = 4              # Number of categorical latent variables.
    classes: int = 4            # Choices per latent variable.
    architecture: str = 'causal-transformer-v1'
    heads: int = 4
    layers: int = 1
    context: int = 16
    unimix: float = .01
    free_nats: float = 1.
    dynamics_scale: float = 1.
    representation_scale: float = .1

    def __post_init__(self):
        for key, low, high in [('deter', 8, 256), ('hidden', 8, 256), ('stoch', 1, 16), ('classes', 2, 32)]:
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError('Invalid latent/head dimensions')
        for key, high in [('unimix', .1), ('free_nats', 20.), ('dynamics_scale', 10.), ('representation_scale', 10.)]:
            value = getattr(self, key)
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= high:
                raise ValueError('Invalid loss/distribution setting')
        if self.architecture != 'causal-transformer-v1':
            raise ValueError('Unknown transformer architecture')
        if (type(self.heads) is not int or not 1 <= self.heads <= 8 or self.deter % self.heads
                or type(self.layers) is not int or not 1 <= self.layers <= 4
                or type(self.context) is not int or not 2 <= self.context <= 128):
            raise ValueError('Invalid bounded transformer dimensions')


@dataclass
class TransformerLatent:
    deter: torch.Tensor         # Latest transformer output.
    stoch: torch.Tensor         # Current categorical state, inferred or predicted.
    history: torch.Tensor
    lengths: torch.Tensor

    def detach(self):
        return TransformerLatent(*(x.detach() for x in
            (self.deter, self.stoch, self.history, self.lengths)))

    def clone(self):
        return TransformerLatent(*(x.clone() for x in
            (self.deter, self.stoch, self.history, self.lengths)))


class CausalMemory(nn.Module):
    def __init__(self, input_width, config):
        super().__init__()
        self.project = nn.Linear(input_width, config.deter)
        self.positions = nn.Parameter(torch.empty(1, config.context, config.deter))
        nn.init.normal_(self.positions, std=.02)
        layer = nn.TransformerEncoderLayer(config.deter, config.heads,
            dim_feedforward=2*config.deter, dropout=0., activation='gelu',
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, config.layers,
            norm=nn.LayerNorm(config.deter), enable_nested_tensor=False)
        # Encoder clones start identical in PyTorch; initialize each separately.
        for block in self.transformer.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            nn.init.zeros_(block.self_attn.in_proj_bias)
            for linear in (block.self_attn.out_proj, block.linear1, block.linear2):
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)

    def forward(self, tokens, lengths):
        width = tokens.shape[1]
        # Right padding keeps the first key valid for every causal query, avoiding
        # all-masked rows/NaNs. No hidden mutable cache or cross-agent attention.
        causal = torch.ones(width, width, dtype=torch.bool, device=tokens.device).triu(1)
        padding = torch.arange(width, device=tokens.device)[None] >= lengths[:, None]
        encoded = self.project(tokens) + self.positions[:, :width]
        return self.transformer(encoded, mask=causal, src_key_padding_mask=padding)


class TransformerWorldModel(nn.Module):
    def __init__(self, spec, config=TransformerConfig()):
        super().__init__()
        if not isinstance(spec, EnvSpec) or not isinstance(config, TransformerConfig):
            raise ValueError('Explicit environment/transformer specifications required')
        self.spec, self.config = spec, config
        self.action_width = spec.action.count if spec.action.kind == 'discrete' else len(spec.action.low)
        if self.action_width > 256:
            raise ValueError('Encoded action width exceeds bound')
        c = config
        self.token_width = c.stoch*c.classes + self.action_width
        def network(inp, out):
            return nn.Sequential(nn.Linear(inp, c.hidden), nn.LayerNorm(c.hidden),
                                 nn.SiLU(), nn.Linear(c.hidden, out))
        self.encoder = network(spec.observation_width, c.hidden)
        self.core = CausalMemory(self.token_width, c)
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
        reference = next(self.parameters())
        return TransformerLatent(reference.new_zeros(batch_size, c.deter),
            reference.new_zeros(batch_size, c.stoch, c.classes),
            reference.new_zeros(batch_size, c.context, self.token_width),
            torch.zeros(batch_size, dtype=torch.long, device=reference.device))

    def _check_state(self, state):
        c = self.config
        if not isinstance(state, TransformerLatent):
            raise ValueError('Transformer history is required; GRU state is incompatible')
        batch = state.deter.shape[0] if state.deter.ndim == 2 else 0
        reference = next(self.parameters())
        if (not 1 <= batch <= 32 or state.deter.shape != (batch, c.deter)
                or state.stoch.shape != (batch, c.stoch, c.classes)
                or any(x.dtype != reference.dtype or x.device != reference.device or not torch.isfinite(x).all()
                       for x in (state.deter, state.stoch))):
            raise ValueError('Invalid latent tensors')
        if (state.history.shape != (batch, c.context, self.token_width)
                or state.history.dtype != state.deter.dtype or state.history.device != state.deter.device
                or not torch.isfinite(state.history).all()
                or state.lengths.shape != (batch,) or state.lengths.dtype != torch.long
                or state.lengths.device != state.deter.device
                or ((state.lengths < 0) | (state.lengths > c.context)).any()):
            raise ValueError('Invalid transformer history or lengths')
        return batch

    def action_features(self, actions):
        if self.spec.action.kind == 'discrete':
            if actions.dtype != torch.long or actions.ndim != 1 or ((actions < 0) | (actions >= self.action_width)).any():
                raise ValueError('Invalid discrete actions')
            return F.one_hot(actions, self.action_width).to(next(self.parameters()).dtype)
        if (actions.dtype != torch.float32 or actions.ndim != 2 or actions.shape[1] != self.action_width
                or not torch.isfinite(actions).all() or (actions.abs() > 1).any()):
            raise ValueError('Invalid normalized continuous actions')
        return actions

    def _advance(self, state, action, is_first):
        batch = self._check_state(state)
        features = self.action_features(action)
        if features.shape[0] != batch or features.device != state.deter.device:
            raise ValueError('Action/state batch or device mismatch')
        if (is_first.dtype != torch.bool or is_first.shape != (batch,)
                or is_first.device != state.deter.device):
            raise ValueError('Exact per-stream reset flag required')
        keep = ~is_first
        lengths = state.lengths * keep.long()
        history = state.history * keep[:, None, None]
        token = torch.cat([state.stoch.flatten(1), features], -1) * keep[:, None]
        context = self.config.context
        shifted = torch.cat([history[:, 1:], torch.zeros_like(history[:, :1])], 1)
        history = torch.where((lengths == context)[:, None, None], shifted, history)
        index = lengths.clamp_max(context-1)
        history = history.scatter(1, index[:, None, None].expand(-1, 1, self.token_width), token[:, None])
        lengths = (lengths+1).clamp_max(context)
        used = int(lengths.max())
        encoded = self.core(history[:, :used], lengths)
        deter = encoded[torch.arange(batch, device=encoded.device), lengths-1]
        return deter, history, lengths

    def _state(self, deter, logits, stochastic, history, lengths):
        p = self.probabilities(logits)
        index = (torch.multinomial(p.flatten(0, 1), 1).reshape(p.shape[:-1])
                 if stochastic else p.argmax(-1))
        hard = F.one_hot(index, self.config.classes).to(p.dtype)
        # Forward: categorical choice. Backward: gradient through probabilities.
        return TransformerLatent(deter, hard + (p - p.detach()), history, lengths)

    def prior_step(self, state, action, *, stochastic=True):
        batch = self._check_state(state)
        deter, history, lengths = self._advance(state, action,
            torch.zeros(batch, dtype=torch.bool, device=state.deter.device))
        logits = self.prior(deter).reshape(batch, self.config.stoch, self.config.classes)
        return self._state(deter, logits, stochastic, history, lengths), logits

    def observe_step(self, state, previous_action, observation, is_first, *, stochastic=True):
        batch = self._check_state(state)
        if (observation.shape != (batch, self.spec.observation_width)
                or observation.dtype != torch.float32 or observation.device != state.deter.device
                or not torch.isfinite(observation).all()):
            raise ValueError('Exact finite float32 observation batch required')
        deter, history, lengths = self._advance(state, previous_action, is_first)
        prior = self.prior(deter).reshape(batch, self.config.stoch, self.config.classes)
        encoded = self.encoder(symlog(observation))
        post = self.posterior(torch.cat([deter, encoded], -1)).reshape_as(prior)
        return self._state(deter, post, stochastic, history, lengths), prior, post

    def features(self, state):
        return torch.cat([state.deter, state.stoch.flatten(1)], -1)

    def heads(self, state, *, decode=True):
        features = self.features(state)
        result = {'reward_symlog': self.reward(features).squeeze(-1),
                  'continuation_logit': self.continuation(features).squeeze(-1)}
        if decode:
            result['observation_symlog'] = self.decoder(features)
        return result

    def probabilities(self, logits):
        c = self.config
        return (1-c.unimix)*logits.softmax(-1) + c.unimix/c.classes

    def balanced_kl(self, posterior, prior):
        q, p = self.probabilities(posterior), self.probabilities(prior)
        tiny = torch.finfo(q.dtype).tiny
        def kl(a, b):
            return (a*(a.clamp_min(tiny).log()-b.clamp_min(tiny).log())).sum((-1, -2))
        # Train predictions toward inference, then inference toward predictions.
        return (kl(q.detach(), p).clamp_min(self.config.free_nats),
                kl(q, p.detach()).clamp_min(self.config.free_nats))


class TransformerModel(nn.Module):
    """World model plus actor/value heads; returns logits and value tensors."""

    def __init__(self, world_model):
        super().__init__()
        if not isinstance(world_model, TransformerWorldModel) or world_model.spec.action.kind != 'discrete':
            raise ValueError('Discrete transformer world model required')
        self.world_model = world_model
        c = world_model.config
        self.actor = nn.Linear(c.deter+c.stoch*c.classes, world_model.spec.action.count)
        self.value = nn.Linear(self.actor.in_features, 1)

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != self.actor.in_features or not torch.isfinite(features).all():
            raise ValueError('Finite posterior features required')
        return self.actor(features), self.value(features).squeeze(-1)
