"""Bounded latent-only MuZero-style MCTS and TD-MPC2-style weighted MPC.

Both planners own a frozen model snapshot per call. No environment interface is
accepted. Planning uses categorical modes, not stochastic/chance-aware search.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import math
import torch
from .policy import sample_action, distribution
from .world_model import symexp


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
