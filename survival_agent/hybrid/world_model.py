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


def sequence_batch(rows, spec):
    """Validate copied real replay rows; stage B requires exact episode-prefix context."""
    if not isinstance(rows, list) or not 1 <= len(rows) <= 32:
        raise ValueError('Bounded replay row list required')
    length = len(rows[0]['actions'])
    if not 1 <= length <= 384 or len(rows)*length > 4096:
        raise ValueError('Sequence exceeds CPU smoke budget')
    packed = {k: [] for k in ['observations', 'actions', 'rewards', 'terminated', 'truncated', 'loss_mask']}
    burns = []
    for row in rows:
        shapes = {'observations': (length+1, spec.observation_width),
                  'actions': (length,) if spec.action.kind == 'discrete' else (length, len(spec.action.low)),
                  'rewards': (length,), 'terminated': (length,), 'truncated': (length,),
                  'loss_mask': (length,), 'is_first': (length+1,)}
        for key, shape in shapes.items():
            value = row[key]
            dtype = np.int64 if key == 'actions' and spec.action.kind == 'discrete' else np.bool_ if key in ['terminated', 'truncated', 'loss_mask', 'is_first'] else np.float32
            if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
                raise ValueError('Exact finite replay shape/dtype required: '+key)
        if row.get('context_is_episode_start') is not True or row.get('episode_offset') != 0 or not row['is_first'][0] or row['is_first'][1:].any():
            raise ValueError('Stage B requires episode-start reconstruction, not approximate interior context')
        if (row['terminated'][:-1] | row['truncated'][:-1]).any():
            raise ValueError('Sequence cannot cross an episode boundary')
        selected = np.flatnonzero(row['loss_mask'])
        if not len(selected) or not np.array_equal(selected, np.arange(selected[0], length)) or selected[0] > 128 or len(selected) > 256:
            raise ValueError('Loss mask must be a bounded burn-in followed by contiguous unroll')
        burns.append(int(selected[0]))
        masks = row.get('masks')
        if masks is not None and (spec.action.kind != 'discrete' or masks.shape != (length+1, spec.action.count) or masks.dtype != np.bool_):
            raise ValueError('Invalid action masks')
        for i, action in enumerate(row['actions']):
            spec.action.validate(action, None if masks is None else masks[i])
        for key in packed:
            packed[key].append(row[key])
    if len(set(burns)) != 1:
        raise ValueError('All rows must share the same burn-in')
    result = {key: torch.from_numpy(np.stack(values).copy()) for key, values in packed.items()}
    # Deliberately ignore any supplied derived continuation: flags are authoritative.
    result['continuation'] = (~result['terminated']).float()
    result['burn_in'] = burns[0]
    return result


def sequence_loss(model, rows, *, stochastic=True):
    batch = sequence_batch(rows, model.spec)
    if next(model.parameters()).device.type != 'cpu':
        raise ValueError('Stage B sequence trainer currently supports CPU only')
    obs, actions = batch['observations'], batch['actions']
    n, length = actions.shape[:2]
    zero_action = torch.zeros_like(actions[:, 0])
    state = model.initial(n)
    burn = batch['burn_in']
    with torch.set_grad_enabled(torch.is_grad_enabled() and burn == 0):
        state, _, _ = model.observe_step(state, zero_action, obs[:, 0], torch.ones(n, dtype=torch.bool), stochastic=stochastic)
    losses = {key: [] for key in ['reconstruction', 'reward', 'continuation', 'dynamics_kl', 'representation_kl']}
    for t in range(length):
        if t == burn:
            state = state.detach() if burn else state
        with torch.set_grad_enabled(torch.is_grad_enabled() and t >= burn):
            state, prior, post = model.observe_step(state, actions[:, t], obs[:, t+1], torch.zeros(n, dtype=torch.bool), stochastic=stochastic)
            if t < burn:
                continue
            head = model.heads(state)
            dyn, rep = model.balanced_kl(post, prior)
            losses['reconstruction'].append((head['observation_symlog']-symlog(obs[:, t+1])).square().mean(-1))
            losses['reward'].append((head['reward_symlog']-symlog(batch['rewards'][:, t])).square())
            losses['continuation'].append(F.binary_cross_entropy_with_logits(head['continuation_logit'], batch['continuation'][:, t], reduction='none'))
            losses['dynamics_kl'].append(dyn)
            losses['representation_kl'].append(rep)
    result = {key: torch.stack(value, 1).mean() for key, value in losses.items()}
    c = model.config
    result['total'] = result['reconstruction']+result['reward']+result['continuation']+c.dynamics_scale*result['dynamics_kl']+c.representation_scale*result['representation_kl']
    if not all(torch.isfinite(value) for value in result.values()):
        raise FloatingPointError('Nonfinite world-model loss')
    return result


class WorldModelTrainer:
    def __init__(self, model, *, learning_rate=3e-4, gradient_clip=10.):
        if type(learning_rate) not in (float, int) or not math.isfinite(learning_rate) or not 0 < learning_rate <= .01 or type(gradient_clip) not in (float, int) or not math.isfinite(gradient_clip) or not 0 < gradient_clip <= 100:
            raise ValueError('Invalid bounded optimizer configuration')
        self.model, self.gradient_clip = model, gradient_clip
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.updates = 0

    def step(self, rows):
        self.optimizer.zero_grad(set_to_none=True)
        losses = sequence_loss(self.model, rows)
        losses['total'].backward()
        norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip, error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        return {**{key: float(value.detach()) for key, value in losses.items()}, 'gradient_norm_before_clip': float(norm)}


@torch.no_grad()
def prediction_diagnostics(model, rows, *, context=1, horizons=(1, 3)):
    """Mode-latent open-loop predictions; only prefix observations infer the start.

    Future observations/rewards are targets only, never fed into prior rollouts.
    Raw errors are descriptive smoke diagnostics, not calibrated likelihoods.
    """
    batch = sequence_batch(rows, model.spec)
    length = batch['actions'].shape[1]
    if type(context) is not int or not 0 <= context < length or not isinstance(horizons, tuple) or not horizons or any(type(h) is not int or not 1 <= h <= min(32, length-context) for h in horizons):
        raise ValueError('Invalid bounded diagnostic context/horizons')
    n = len(rows)
    state, _, _ = model.observe_step(model.initial(n), torch.zeros_like(batch['actions'][:, 0]), batch['observations'][:, 0], torch.ones(n, dtype=torch.bool), stochastic=False)
    for t in range(context):
        state, _, _ = model.observe_step(state, batch['actions'][:, t], batch['observations'][:, t+1], torch.zeros(n, dtype=torch.bool), stochastic=False)
    report = {}
    for h in range(1, max(horizons)+1):
        t = context+h-1
        state, _ = model.prior_step(state, batch['actions'][:, t], stochastic=False)
        if h in horizons:
            heads = model.heads(state)
            predicted_obs, predicted_reward = symexp(heads['observation_symlog']), symexp(heads['reward_symlog'])
            values = {'observation_mse': (predicted_obs-batch['observations'][:, t+1]).square().mean(),
                      'reward_mse': (predicted_reward-batch['rewards'][:, t]).square().mean(),
                      'continuation_bce': F.binary_cross_entropy_with_logits(heads['continuation_logit'], batch['continuation'][:, t])}
            if not all(torch.isfinite(v) for v in values.values()):
                raise FloatingPointError('Nonfinite prediction diagnostics')
            report[str(h)] = {k: float(v) for k, v in values.items()}
    return report
