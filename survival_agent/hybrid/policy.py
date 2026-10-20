"""Action distributions, separate from neural architecture and environment I/O."""
import math
import torch
from torch.nn import functional as F


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
