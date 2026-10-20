"""One-file Transformer / Dreamer / MuZero / TD-MPC2 experimental hybrid.

Read top to bottom: neural architecture, action distributions, latent planners,
replay, world-model fitting, imagination/TD learning, controller, training loop.
Run with `python -m survival_agent.hybrid.model --help`. Environment definitions
and the shared safe checkpoint serializer are external; no environment is loaded
and no training starts on import. Planning never executes environment actions.
"""
from copy import deepcopy
from dataclasses import dataclass, field, asdict
import math
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .contracts import EnvSpec, ActionSpec, VectorAdapter, flag, vector


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


@dataclass(frozen=True)
class HybridConfig:
    """Additional neural heads; Transformer dimensions live in TransformerConfig."""
    hidden: int = 64
    q_ensemble: int = 5
    min_std: float = .1
    max_std: float = 1.

    def __post_init__(self):
        if type(self.hidden) is not int or not 8 <= self.hidden <= 256:
            raise ValueError('Hidden width must be 8..256')
        if type(self.q_ensemble) is not int or not 2 <= self.q_ensemble <= 10:
            raise ValueError('Q ensemble must contain 2..10 heads')
        if not 0 < self.min_std <= self.max_std <= 2:
            raise ValueError('Invalid policy standard deviation bounds')


class HybridModel(nn.Module):
    """Shared Transformer/Dreamer model serving MuZero and TD-MPC2-style search.

    All neural layers are here. Discrete policy logits and state values support
    MCTS; continuous Gaussian parameters and separate Q heads support MPC/TD.
    MCTS, MPC and the complete training loop are defined below in this file.
    Scalar value/Q predictions use raw reward units; reward reconstruction uses
    symlog. This is an experimental hybrid, not any full reference algorithm.
    """
    def __init__(self, spec, transformer=TransformerConfig(), config=HybridConfig()):
        super().__init__()
        self.world_model = TransformerWorldModel(spec, transformer)
        self.spec, self.config = spec, config
        width = transformer.deter + transformer.stoch * transformer.classes
        def head(inp, out):
            return nn.Sequential(nn.Linear(inp, config.hidden), nn.LayerNorm(config.hidden),
                                 nn.SiLU(), nn.Linear(config.hidden, out))
        self.feature_width = width
        self.action_width = self.world_model.action_width
        self.actor = head(width, self.action_width * (2 if spec.action.kind == 'continuous' else 1))
        self.value = head(width, 1)
        self.q_ensemble = nn.ModuleList(
            head(width + self.action_width, 1) for _ in range(config.q_ensemble)
        ) if spec.action.kind == 'continuous' else nn.ModuleList()

    def policy(self, features):
        parameters = self.actor(features)
        if self.spec.action.kind == 'discrete':
            return {'logits': parameters}
        mean, raw_std = parameters.chunk(2, -1)
        std = self.config.min_std + (self.config.max_std-self.config.min_std) * raw_std.sigmoid()
        return {'mean': mean, 'std': std}  # Tanh squashing belongs to the distribution adapter.

    def q_values(self, features, actions):
        if self.spec.action.kind != 'continuous':
            raise ValueError('Q ensemble is specific to continuous control')
        self.world_model.action_features(actions)
        inputs = torch.cat([features, actions], -1)
        return torch.stack([head(inputs).squeeze(-1) for head in self.q_ensemble])

    def forward(self, features, actions=None):
        result = {**self.policy(features), 'value': self.value(features).squeeze(-1)}
        if actions is not None:
            result['q_values'] = self.q_values(features, actions)
        return result


# Action distributions used by the model's planners.

def distribution(model, features, mask=None):
    p = model.policy(features)
    if 'logits' in p:
        logits = p['logits']
        if mask is not None:
            mask = torch.as_tensor(mask, device=logits.device)
            if mask.dtype != torch.bool or mask.shape != logits.shape or not mask.any(-1).all():
                raise ValueError('A boolean mask with at least one legal action per row is required')
            logits = logits.masked_fill(~mask, -torch.inf)
        return torch.distributions.Categorical(logits=logits)
    if mask is not None:
        raise ValueError('Continuous policies do not accept action masks')
    return torch.distributions.Independent(torch.distributions.Normal(p['mean'], p['std']), 1)


def sample_action(model, features, *, deterministic=False, mask=None):
    dist = distribution(model, features, mask)
    if model.spec.action.kind == 'discrete':
        return dist.probs.argmax(-1) if deterministic else dist.sample()
    return (dist.base_dist.loc if deterministic else dist.sample()).tanh()



# Latent planning: no environment calls or live-state mutation.

def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


@dataclass(frozen=True)
class PlannerConfig:
    gamma: float = .99
    horizon: int = 5
    simulations: int = 32
    pb_c_base: float = 19652.
    pb_c_init: float = 1.25
    root_noise: float = .25
    dirichlet_alpha: float = .3
    temperature: float = 1.
    candidates: int = 32
    actor_candidates: int = 4
    elites: int = 8
    iterations: int = 4
    score_scale: float = .5
    min_std: float = .05
    max_std: float = 1.
    terminal_value: str = 'q'

    def __post_init__(self):
        for key, lo, hi in [('horizon',1,32),('simulations',1,1024),('candidates',2,32),
                            ('actor_candidates',1,32),('elites',1,32),('iterations',1,20)]:
            if type(getattr(self,key)) is not int or not lo <= getattr(self,key) <= hi:
                raise ValueError('Invalid planner budget: '+key)
        if not self.actor_candidates < self.candidates or self.elites > self.candidates:
            raise ValueError('Invalid candidate proportions')
        for name in ('gamma','pb_c_base','pb_c_init','root_noise','dirichlet_alpha',
                     'temperature','score_scale','min_std','max_std'):
            if not math.isfinite(getattr(self,name)):
                raise ValueError('Nonfinite planner configuration')
        if not (0 <= self.gamma <= 1 and 0 <= self.root_noise <= 1 and self.pb_c_base > 0
                and self.pb_c_init > 0 and self.dirichlet_alpha > 0 and self.temperature > 0
                and self.score_scale > 0 and 0 < self.min_std <= self.max_std <= 2
                and self.terminal_value in ('q','value')):
            raise ValueError('Invalid planner configuration')


@dataclass
class Plan:
    action: torch.Tensor
    target: torch.Tensor
    diagnostics: dict
    warm_start: object = None


def frozen_snapshot(model):
    return deepcopy(model).eval().requires_grad_(False)


def repeat_state(state, count):
    return type(state)(*(x.repeat_interleave(count, dim=0) for x in
                        (state.deter, state.stoch, state.history, state.lengths)))


def transition(model, state, action):
    next_state, _ = model.world_model.prior_step(state, action, stochastic=False)
    heads = model.world_model.heads(next_state, decode=False)
    reward = symexp(heads['reward_symlog'])
    continuation = heads['continuation_logit'].sigmoid()
    if not torch.isfinite(reward).all() or not torch.isfinite(continuation).all():
        raise FloatingPointError('Nonfinite imagined transition')
    return next_state, reward, continuation


@dataclass
class Node:
    state: object
    prior: torch.Tensor
    value: float
    visits: torch.Tensor
    totals: torch.Tensor
    children: dict = field(default_factory=dict)


@torch.no_grad()
def mcts(model, state, config=PlannerConfig(), *, mask=None, training=False):
    if model.spec.action.kind != 'discrete' or state.deter.shape[0] != 1:
        raise ValueError('MCTS expects one discrete inferred state')
    model = frozen_snapshot(model)
    device = state.deter.device
    count = model.spec.action.count
    root_mask = torch.ones(1,count,dtype=torch.bool,device=device) if mask is None else torch.as_tensor(mask, device=device)
    if root_mask.shape == (count,):
        root_mask = root_mask[None]
    def node(latent, legal=None):
        features = model.world_model.features(latent)
        prior = distribution(model,features,legal).probs[0]
        value = float(model.value(features).squeeze())
        if not math.isfinite(value):
            raise FloatingPointError('Nonfinite search value')
        return Node(latent, prior, value, torch.zeros(count,device=device), torch.zeros(count,device=device))
    root = node(state.detach().clone(), root_mask)
    if training and config.root_noise:
        noise = torch.distributions.Dirichlet(torch.full((int(root_mask.sum()),),config.dirichlet_alpha,device=device)).sample()
        root.prior[root_mask[0]] = (1-config.root_noise)*root.prior[root_mask[0]] + config.root_noise*noise
    low, high, calls = math.inf, -math.inf, 0
    for _ in range(config.simulations):
        current, path = root, []
        leaf_value = current.value
        for depth in range(config.horizon):
            total = float(current.visits.sum())
            average = current.totals / current.visits.clamp_min(1)
            normalized = (average-low)/(high-low) if high > low else torch.zeros_like(average)
            normalized = torch.where(current.visits > 0, normalized, torch.zeros_like(normalized))
            c = math.log((total+config.pb_c_base+1)/config.pb_c_base)+config.pb_c_init
            scores = normalized + c * current.prior * math.sqrt(total+1)/(1+current.visits)
            if current is root:
                scores = scores.masked_fill(~root_mask[0], -torch.inf)
            action = int(scores.argmax())
            fresh = action not in current.children
            if fresh:
                latent,reward,continuation = transition(model,current.state,torch.tensor([action],device=device))
                current.children[action] = (node(latent),float(reward),float(continuation))
                calls += 1
            child,reward,continuation = current.children[action]
            path.append((current,action,reward,continuation))
            current = child
            leaf_value = child.value
            if fresh or continuation == 0:
                break
        for parent,action,reward,continuation in reversed(path):
            leaf_value = reward + config.gamma*continuation*leaf_value
            parent.visits[action] += 1
            parent.totals[action] += leaf_value
            q = float(parent.totals[action]/parent.visits[action])
            low,high = min(low,q),max(high,q)
    target = root.visits/root.visits.sum()
    if training:
        probabilities = (root.visits.clamp_min(1e-30).log()/config.temperature).softmax(-1)
        probabilities = probabilities.masked_fill(root.visits == 0, 0)
        action = torch.multinomial(probabilities,1)
    else:
        action = root.visits.argmax().reshape(1)
    return Plan(action.detach(),target[None].detach(),
                {'planner':'mcts','simulations':config.simulations,'model_calls':calls,
                 'visits':root.visits.tolist(),'latent_approximation':'categorical_mode',
                 'interior_legality':'unmasked'})


@torch.no_grad()
def mpc(model, state, config=PlannerConfig(), *, warm_start=None, training=False):
    if model.spec.action.kind != 'continuous' or state.deter.shape[0] != 1:
        raise ValueError('MPC expects one continuous inferred state')
    model = frozen_snapshot(model)
    c, device = config,state.deter.device
    shape = (c.horizon,model.action_width)
    mean = torch.zeros(shape,device=device)
    if warm_start is not None:
        if warm_start.shape != shape or not torch.isfinite(warm_start).all() or (warm_start.abs()>1).any():
            raise ValueError('Invalid previous plan')
        mean[:-1] = warm_start.detach()[1:]
    std = torch.full_like(mean,c.max_std)
    proposal_state = repeat_state(state.detach().clone(),c.actor_candidates)
    proposals = []
    for _ in range(c.horizon):
        action = sample_action(model,model.world_model.features(proposal_state))
        proposals.append(action)
        proposal_state,_,_ = transition(model,proposal_state,action)
    proposals = torch.stack(proposals,1)
    # Reference uses the mean of two subsampled online Qs for planning; keep
    # the pair fixed during this search to avoid candidate-dependent noise.
    pair = torch.randperm(len(model.q_ensemble),device=device)[:2]
    calls = c.horizon
    for _ in range(c.iterations):
        candidates = (mean[None]+std[None]*torch.randn(c.candidates,*shape,device=device)).clamp(-1,1)
        candidates[:c.actor_candidates] = proposals
        latent = repeat_state(state.detach().clone(),c.candidates)
        score = torch.zeros(c.candidates,device=device)
        discount = torch.ones_like(score)
        for t in range(c.horizon):
            latent,reward,continuation = transition(model,latent,candidates[:,t])
            score += discount*reward
            discount *= c.gamma*continuation
            calls += 1
        features = model.world_model.features(latent)
        if c.terminal_value == 'q':
            action = sample_action(model,features,deterministic=True)
            tail = model.q_values(features,action)[pair].mean(0)
        else:
            tail = model.value(features).squeeze(-1)
        score += discount*tail
        if not torch.isfinite(score).all():
            raise FloatingPointError('Nonfinite MPC score')
        index = score.topk(c.elites).indices
        elites,values = candidates[index],score[index]
        weights = (c.score_scale*(values-values.max())).softmax(0)
        mean = (weights[:,None,None]*elites).sum(0).clamp(-1,1)
        std = (weights[:,None,None]*(elites-mean[None]).square()).sum(0).sqrt().clamp(c.min_std,c.max_std)
    # Evaluation is deterministic given RNG state: select the best scored elite.
    selected = int(torch.multinomial(weights,1)) if training else 0
    action = elites[selected,0]
    if training:
        action = action+std[0]*torch.randn_like(action)
    return Plan(action.clamp(-1,1)[None].detach(),mean[0:1].detach(),
                {'planner':'mpc','model_calls':calls,'iterations':c.iterations,
                 'candidates':c.candidates,'best_predicted_return':float(values[0]),
                 'terminal_value':c.terminal_value,'latent_approximation':'categorical_mode'},mean.detach())


# Policy log probabilities for imagination learning.

def log_probability(model, features, actions, mask=None):
    dist = distribution(model, features, mask)
    if model.spec.action.kind == 'discrete':
        return dist.log_prob(actions.detach()), dist.entropy()
    # Score-function objective: sampled actions are constants, never rsample paths.
    raw = torch.atanh(actions.detach().clamp(-1+1e-6, 1-1e-6))
    correction = (2*(math.log(2)-raw-F.softplus(-2*raw))).sum(-1)
    logp = dist.log_prob(raw) - correction
    # Reparameterized Monte Carlo entropy for the squashed distribution only.
    z = dist.rsample()
    entropy = -dist.log_prob(z) + (2*(math.log(2)-z-F.softplus(-2*z))).sum(-1)
    return logp, entropy


# Real-experience replay; normalized actions and complete episode boundaries.

FIELDS = {'observations', 'actions', 'rewards', 'terminated', 'truncated', 'is_first', 'masks'}


class SequenceReplay:
    def __init__(self, spec, *, seed=0, capacity=4096, max_bytes=8*1024*1024):
        if not isinstance(spec, EnvSpec) or type(capacity) is not int or not 1 <= capacity <= 16384:
            raise ValueError('Invalid bounded replay capacity')
        if type(max_bytes) is not int or not 1024 <= max_bytes <= 16*1024*1024:
            raise ValueError('Invalid bounded replay byte budget')
        self.spec, self.capacity, self.max_bytes = spec, capacity, max_bytes
        self.rng = np.random.default_rng(seed)
        self.episodes = []
        self.pending = None

    def start(self, observation, mask=None):
        if self.pending is not None:
            raise RuntimeError('Cannot reset an unfinished replay episode')
        observation = self.spec.observation(observation)
        mask = self.spec.action.mask(mask)
        self.pending = {'observations': [observation], 'actions': [], 'rewards': [],
                        'terminated': [], 'truncated': [], 'masks': [mask]}

    def append(self, action, reward, next_observation, terminated, truncated, next_mask=None):
        if self.pending is None:
            raise RuntimeError('Start an episode before appending')
        episode = self.pending
        terminal, truncation = flag(terminated), flag(truncated)
        observation = self.spec.observation(next_observation)
        mask = self.spec.action.mask(next_mask)
        if (mask is None) != (episode['masks'][0] is None):
            raise ValueError('Mask availability must be consistent within an episode')
        action = self.spec.action.validate(action, episode['masks'][-1])
        if isinstance(reward, (bool, np.bool_)) or not isinstance(reward, (int, float, np.integer, np.floating)) or not np.isfinite(reward):
            raise ValueError('Finite scalar reward required')
        with np.errstate(over='ignore'):
            reward32 = np.float32(reward)
        if not np.isfinite(reward32):
            raise ValueError('Reward not representable in float32')
        n = len(episode['actions']) + 1
        action_width = 1 if self.spec.action.kind == 'discrete' else len(self.spec.action.low)
        estimate = (n+1)*self.spec.observation_width*4 + n*(action_width*8+6) + n+1
        if mask is not None:
            estimate += (n+1)*self.spec.action.count
        if n > min(4096, self.capacity) or estimate > self.max_bytes:
            raise ValueError('Episode exceeds replay bounds; collector must use a shorter horizon')
        episode['actions'].append(action)
        episode['rewards'].append(float(reward32))
        episode['terminated'].append(terminal)
        episode['truncated'].append(truncation)
        episode['observations'].append(observation)
        episode['masks'].append(mask)
        if terminal or truncation:
            packed = {key: np.asarray(episode[key], dtype=np.float32 if key in ['observations','rewards'] else
                                      np.int64 if key == 'actions' and self.spec.action.kind == 'discrete' else
                                      np.float32 if key == 'actions' else bool)
                      for key in ['observations','actions','rewards','terminated','truncated']}
            packed['is_first'] = np.zeros(n+1, bool)
            packed['is_first'][0] = True
            packed['masks'] = None if mask is None else np.asarray(episode['masks'], bool)
            self.validate_episode(packed)
            self.episodes.append(packed)
            self.pending = None
            while sum(len(e['actions']) for e in self.episodes) > self.capacity or sum(self.nbytes(e) for e in self.episodes) > self.max_bytes:
                self.episodes.pop(0)

    @staticmethod
    def nbytes(episode):
        return sum(value.nbytes for value in episode.values() if isinstance(value, np.ndarray))

    def validate_episode(self, episode):
        if not isinstance(episode, dict) or set(episode) != FIELDS:
            raise ValueError('Invalid replay episode schema')
        if any(not isinstance(episode[key], np.ndarray) for key in FIELDS - {'masks'}):
            raise ValueError('Replay fields must be arrays')
        actions = episode['actions']
        n = actions.shape[0] if actions.ndim else 0
        if not 1 <= n <= min(4096, self.capacity) or self.nbytes(episode) > self.max_bytes:
            raise ValueError('Oversize replay episode')
        width = self.spec.observation_width
        if episode['observations'].shape != (n+1, width) or episode['observations'].dtype != np.float32:
            raise ValueError('Invalid observation sequence')
        for key in ['terminated', 'truncated']:
            if episode[key].shape != (n,) or episode[key].dtype != np.bool_:
                raise ValueError('Exact boolean transition shape required')
        if episode['rewards'].shape != (n,) or episode['rewards'].dtype != np.float32:
            raise ValueError('Exact float reward shape required')
        if episode['is_first'].shape != (n+1,) or episode['is_first'].dtype != np.bool_ or not episode['is_first'][0] or episode['is_first'][1:].any():
            raise ValueError('Exactly one episode start required')
        boundary = episode['terminated'] | episode['truncated']
        if not boundary[-1] or boundary[:-1].any():
            raise ValueError('Complete episodes cannot cross boundaries')
        expected = (n,) if self.spec.action.kind == 'discrete' else (n,len(self.spec.action.low))
        dtype = np.int64 if self.spec.action.kind == 'discrete' else np.float32
        if actions.shape != expected or actions.dtype != dtype:
            raise ValueError('Invalid canonical action sequence')
        for value in [episode['observations'], episode['rewards'], actions]:
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite replay value')
        masks = episode['masks']
        if masks is not None:
            if self.spec.action.kind != 'discrete' or not isinstance(masks, np.ndarray) or masks.shape != (n+1,self.spec.action.count) or masks.dtype != np.bool_:
                raise ValueError('Exact boolean mask sequence required')
        for index, action in enumerate(actions):
            self.spec.action.validate(action, None if masks is None else masks[index])

    def sample(self, batch_size=1, *, burn_in=0, unroll=1, require_episode_start=True):
        if any(type(x) is not int for x in [batch_size,burn_in,unroll]) or not 1 <= batch_size <= 32 or not 0 <= burn_in <= 128 or not 1 <= unroll <= 256:
            raise ValueError('Invalid bounded sequence request')
        if type(require_episode_start) is not bool:
            raise ValueError('Declare reconstruction context explicitly')
        length = burn_in + unroll
        candidates = [(e, 1 if require_episode_start else len(e['actions'])-length+1)
                      for e in self.episodes if len(e['actions']) >= length]
        if not candidates:
            raise ValueError('No complete episode long enough; sequences are never joined or padded')
        total = sum(count for _,count in candidates)
        rows = []
        for _ in range(batch_size):
            choice = int(self.rng.integers(total))
            for episode, count in candidates:
                if choice < count:
                    break
                choice -= count
            start = 0 if require_episode_start else choice
            end = start + length
            row = {key: None if value is None else value[start:end+(key in ['observations','is_first','masks'])].copy()
                   for key,value in episode.items()}
            row['loss_mask'] = np.arange(length) >= burn_in
            row['continuation'] = (~row['terminated']).astype(np.float32)
            row['context_is_episode_start'] = start == 0
            row['episode_offset'] = start
            rows.append(row)
        return rows

    def state_dict(self):
        if self.pending is not None:
            raise ValueError('Stage-A checkpointing is at episode boundaries only')
        return deepcopy({'version':1, 'spec':asdict(self.spec), 'capacity':self.capacity,
                         'max_bytes':self.max_bytes, 'rng':self.rng.bit_generator.state, 'episodes':self.episodes})

    @classmethod
    def from_state_dict(cls, state, expected_spec):
        if not isinstance(state, dict) or set(state) != {'version','spec','capacity','max_bytes','rng','episodes'} or state['version'] != 1 or state['spec'] != asdict(expected_spec):
            raise ValueError('Replay version/spec mismatch')
        replay = cls(expected_spec, capacity=state['capacity'], max_bytes=state['max_bytes'])
        episodes = state['episodes']
        if not isinstance(episodes, list) or len(episodes) > replay.capacity:
            raise ValueError('Invalid episode collection')
        # Check totals before copying, and validate shapes before boolean arithmetic.
        for episode in episodes:
            replay.validate_episode(episode)
        if sum(len(e['actions']) for e in episodes) > replay.capacity or sum(replay.nbytes(e) for e in episodes) > replay.max_bytes:
            raise ValueError('Replay exceeds total bounds')
        replay.episodes = deepcopy(episodes)
        replay.rng.bit_generator.state = deepcopy(state['rng'])
        return replay


# World-model sequence fitting and prediction diagnostics.

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


# Dreamer-style imagination, real TD critics and optional planner distillation.

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


# Interaction state and actor/planner dispatch.

class HybridController:
    def __init__(self, model, *, mode='auto', planning=True, config=PlannerConfig()):
        if mode not in ('auto','actor','mcts','mpc'):
            raise ValueError('Unknown control mode')
        selected = model.spec.action.planner_name(planning) if mode == 'auto' else mode
        if (selected == 'mcts' and model.spec.action.kind != 'discrete'
                or selected == 'mpc' and model.spec.action.kind != 'continuous'):
            raise ValueError('Planner/action-space mismatch')
        self.model,self.mode,self.config = model,selected,config
        self.states,self.previous_actions,self.warm_starts = {},{},{}

    def reset(self, stream=0):
        for slots in (self.states,self.previous_actions,self.warm_starts):
            slots.pop(stream,None)

    def boundary(self, *, terminated, truncated, stream=0):
        if type(terminated) is not bool or type(truncated) is not bool:
            raise ValueError('Explicit episode flags required')
        if terminated or truncated:
            self.reset(stream)

    @torch.no_grad()
    def decide(self, observation, *, stream=0, mask=None, training=False):
        if type(stream) is not int or not 0 <= stream < 1024:
            raise ValueError('Invalid stream')
        model,world = self.model,self.model.world_model
        device = next(model.parameters()).device
        obs = torch.as_tensor(model.spec.observation(observation),device=device)[None]
        first = stream not in self.states
        previous = self.previous_actions.get(stream)
        if first:
            previous = torch.zeros(1,dtype=torch.long,device=device) if model.spec.action.kind == 'discrete' else torch.zeros(1,model.action_width,device=device)
        state,_,_ = world.observe_step(self.states.get(stream,world.initial(1)),previous,obs,
                                       torch.tensor([first],device=device),stochastic=False)
        if self.mode == 'mcts':
            plan = mcts(model,state,self.config,mask=mask,training=training)
        elif self.mode == 'mpc':
            if mask is not None:
                raise ValueError('Continuous control does not accept action masks')
            plan = mpc(model,state,self.config,warm_start=self.warm_starts.get(stream),training=training)
        else:
            legal = None if mask is None else torch.as_tensor(mask,device=device)[None]
            action = sample_action(model,world.features(state),deterministic=not training,mask=legal)
            plan = Plan(action.detach(),action.detach(),{'planner':'actor'})
        self.states[stream],self.previous_actions[stream] = state.detach(),plan.action.detach()
        if plan.warm_start is not None:
            self.warm_starts[stream] = plan.warm_start.clone()
        return plan


# Full training loop: collect -> replay -> world -> imagination -> TD -> evaluate.

@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20261021
    cycles: int = 4
    initial_episodes: int = 2
    episodes_per_cycle: int = 1
    max_episode_steps: int = 16
    updates_per_cycle: int = 2
    batch_size: int = 1
    max_prefix: int = 16
    replay_capacity: int = 4096
    mode: str = 'auto'
    evaluation_episodes: int = 1

    def __post_init__(self):
        bounds = {'seed':(0,2**31-1),'cycles':(1,10000),'initial_episodes':(1,100),
                  'episodes_per_cycle':(1,100),'max_episode_steps':(1,128),
                  'updates_per_cycle':(1,1000),'batch_size':(1,32),'max_prefix':(1,128),
                  'replay_capacity':(128,16384),'evaluation_episodes':(0,100)}
        for name,(low,high) in bounds.items():
            if type(getattr(self,name)) is not int or not low <= getattr(self,name) <= high:
                raise ValueError('Invalid training configuration: '+name)
        if self.mode not in ('actor','auto','mcts','mpc'):
            raise ValueError('Unknown training control mode')


class TrainingSession:
    """An episode-boundary resumable CPU training loop using a supplied environment.

    Environment physics stay outside this file. Collection, replay, all parameter
    updates, evaluation and checkpoint orchestration are implemented here.
    No environment or training process is started by constructing a session.
    """
    def __init__(self, spec, training=TrainingConfig(), transformer=TransformerConfig(),
                 heads=HybridConfig(), learning=LearningConfig(), planner=PlannerConfig()):
        if learning.gamma != planner.gamma:
            raise ValueError('Training and planning must use the same discount')
        self.spec,self.config = spec,training
        self.model = HybridModel(spec,transformer,heads)
        self.learner = HybridLearner(self.model,learning)
        self.controller = HybridController(self.model,mode=training.mode,config=planner)
        self.replay = SequenceReplay(spec,seed=training.seed,capacity=training.replay_capacity)
        self.rng = np.random.default_rng(training.seed)
        self.episodes,self.transitions,self.cycles = 0,0,0

    def configuration(self):
        return {'format':'hybrid-one-file-v1','spec':asdict(self.spec),
                'training':asdict(self.config),'transformer':asdict(self.model.world_model.config),
                'heads':asdict(self.model.config),'learning':asdict(self.learner.config),
                'planner':asdict(self.controller.config)}

    def collect_episode(self, env, *, exploratory=False):
        """The only training method that executes real environment transitions."""
        adapter = VectorAdapter(env,self.spec)
        observation,info = adapter.reset(seed=self.config.seed+self.episodes)
        mask = adapter.current_mask
        self.controller.reset()
        self.replay.start(observation,mask)
        total = 0.
        for t in range(self.config.max_episode_steps):
            if exploratory:
                if self.spec.action.kind == 'discrete':
                    legal = np.arange(self.spec.action.count) if mask is None else np.flatnonzero(mask)
                    if not len(legal):
                        raise ValueError('No legal exploration action')
                    action = int(self.rng.choice(legal))
                else:
                    action = self.rng.uniform(-1,1,len(self.spec.action.low)).astype(np.float32)
            else:
                plan = self.controller.decide(observation,mask=mask,training=True)
                action = int(plan.action.item()) if self.spec.action.kind == 'discrete' else plan.action[0].cpu().numpy()
            item = adapter.step(action)
            # Our collection limit is an external truncation, never a termination.
            truncated = item.truncated or (t+1 == self.config.max_episode_steps and not item.terminated)
            self.replay.append(action,item.reward,item.next_observation,item.terminated,truncated,item.next_mask)
            total += item.reward
            self.transitions += 1
            self.controller.boundary(terminated=item.terminated,truncated=truncated)
            observation,mask = item.next_observation,item.next_mask
            if item.terminated or truncated:
                break
        self.episodes += 1
        return {'return':total,'steps':t+1,'exploratory':exploratory}

    def update(self):
        if self.replay.pending is not None or not self.replay.episodes:
            raise ValueError('Training requires complete real replay episodes')
        length = int(self.rng.integers(1,min(self.config.max_prefix,
                         max(len(e['actions']) for e in self.replay.episodes))+1))
        rows = self.replay.sample(self.config.batch_size,burn_in=length-1,unroll=1)
        metrics = {'world':self.learner.world_step(rows)}
        teacher = None
        if self.learner.config.distillation_weight:
            targets = []
            name = self.spec.action.planner_name(True)
            for row in rows:
                state,_,_ = replay_states(self.model,[row])
                if name == 'mcts':
                    mask = None if row['masks'] is None else row['masks'][-2]
                    result = mcts(self.model,state,self.controller.config,mask=mask)
                else:
                    result = mpc(self.model,state,self.controller.config)
                targets.append(result.target)
            plan = Plan(result.action,torch.cat(targets),{'planner':name})
            teacher = planner_target(rows,plan,version=self.learner.model_version,
                                     step=self.learner.updates,settings=asdict(self.controller.config))
        metrics['imagination'] = self.learner.imagination_step(rows,teacher)
        if self.spec.action.kind == 'continuous':
            metrics['td'] = self.learner.td_step(rows)
        return metrics

    def evaluate(self, env):
        """Fixed-seed actor/planner evaluations, without updates or training RNG drift."""
        result = {}
        modes = list(dict.fromkeys(('actor',self.controller.mode)))
        with torch.random.fork_rng(devices=[]):
            for mode in modes:
                returns = []
                for episode in range(self.config.evaluation_episodes):
                    seed = self.config.seed+1000000+episode
                    torch.manual_seed(seed)
                    controller = HybridController(self.model,mode=mode,config=self.controller.config)
                    adapter = VectorAdapter(env,self.spec)
                    observation,_ = adapter.reset(seed=seed)
                    total = 0.
                    for _ in range(self.config.max_episode_steps):
                        plan = controller.decide(observation,mask=adapter.current_mask)
                        action = int(plan.action.item()) if self.spec.action.kind == 'discrete' else plan.action[0].cpu().numpy()
                        item = adapter.step(action)
                        total += item.reward
                        observation = item.next_observation
                        if item.terminated or item.truncated:
                            break
                    returns.append(total)
                result[mode] = returns
        return result

    def save(self, path):
        # Shared serializer is deliberately pickle-free and source/version checked.
        from . import checkpoints
        checkpoints.save(path,self.replay,components=self.learner.state_dict(),
                         config=self.configuration(),rng=self.rng,
                         counters={'episodes':self.episodes,'transitions':self.transitions,'cycles':self.cycles})

    def restore(self, path):
        from . import checkpoints
        replay,components,counters = checkpoints.load(path,self.spec,config=self.configuration(),rng=self.rng)
        self.learner.load_state_dict(components)
        self.replay = replay
        self.episodes,self.transitions,self.cycles = (counters[k] for k in ('episodes','transitions','cycles'))
        self.controller.reset()

    def run(self, env, *, output=None, max_cycles=None):
        """Collect and optimize repeatedly; optional limit pauses at a safe boundary."""
        import json
        from pathlib import Path
        if max_cycles is not None and (type(max_cycles) is not int or max_cycles < 0):
            raise ValueError('Invalid cycle limit')
        destination = None
        if output is not None:
            destination = Path(output)
            destination.mkdir(parents=True,exist_ok=False,mode=0o700)
            (destination/'config.json').write_text(json.dumps(self.configuration(),indent=2)+'\n')
        while self.episodes < self.config.initial_episodes:
            self.collect_episode(env,exploratory=True)
        logs = []
        end = self.config.cycles if max_cycles is None else min(self.config.cycles,self.cycles+max_cycles)
        while self.cycles < end:
            # Bootstrap policy/planning from fitted real data before collecting
            # the next episode, then repeat the same collection/update cycle.
            updates = [self.update() for _ in range(self.config.updates_per_cycle)]
            collected = [self.collect_episode(env) for _ in range(self.config.episodes_per_cycle)]
            self.cycles += 1
            record = {'cycle':self.cycles,'updates':updates,'collected':collected,
                      'evaluation':self.evaluate(env) if self.config.evaluation_episodes else {}}
            logs.append(record)
            if destination:
                with (destination/'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(record,allow_nan=False)+'\n')
                self.save(destination/'checkpoint.safe')
        result = {'cycles':self.cycles,'episodes':self.episodes,'transitions':self.transitions,
                  'logs':logs,'claim':'Engineering run; no learning-improvement or retention claim'}
        if destination:
            if not logs:
                self.save(destination/'checkpoint.safe')
            (destination/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            for path in destination.iterdir():
                path.chmod(0o600)
        return result


def main():
    """Executable CPU training entry point, kept beside the complete model."""
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['CartPole-v1','Pendulum-v1'],default='CartPole-v1')
    parser.add_argument('--output',required=True,help='New directory for logs and a safe checkpoint')
    parser.add_argument('--resume',help='Episode-boundary checkpoint with matching configuration')
    parser.add_argument('--cycles',type=int,default=4)
    parser.add_argument('--run-cycles',type=int,help='Run only this many additional cycles, then save')
    parser.add_argument('--seed',type=int,default=20261021)
    parser.add_argument('--episode-steps',type=int,default=16)
    parser.add_argument('--updates-per-cycle',type=int,default=2)
    parser.add_argument('--mode',choices=['auto','actor','mcts','mpc'],default='auto')
    parser.add_argument('--distillation',action='store_true')
    args = parser.parse_args()
    # Optional environment dependency is imported only when the CLI is invoked.
    import gymnasium as gym
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    training = TrainingConfig(seed=args.seed,cycles=args.cycles,max_episode_steps=args.episode_steps,
                              updates_per_cycle=args.updates_per_cycle,mode=args.mode)
    env = gym.make(args.task)
    try:
        session = TrainingSession(EnvSpec.from_gymnasium(env),training,
            learning=LearningConfig(distillation_weight=.1 if args.distillation else 0.))
        if args.resume:
            session.restore(args.resume)
        result = session.run(env,output=args.output,max_cycles=args.run_cycles)
        print(json.dumps({k:result[k] for k in ('cycles','episodes','transitions','claim')}))
    finally:
        env.close()


if __name__ == '__main__':
    main()
