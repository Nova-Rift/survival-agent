"""Bounded posterior/prior diagnostics; never environment calls or control claims."""
import numpy as np
import torch
from .world_model import sequence_batch, symexp


@torch.no_grad()
def episode_predictions(model, row, *, context=1):
    """Infer o[0:context+1], then fork separate prior-only and teacher-forced paths.

    Returns transition-indexed predictions, not rewards/continuations from a reset.
    Categorical modes make these deterministic approximations, not belief integrals.
    """
    batch = sequence_batch([row], model.spec)
    length = batch['actions'].shape[1]
    if type(context) is not int or not 0 <= context < length or length-context > 32:
        raise ValueError('Invalid bounded diagnostic context')
    if not bool(batch['terminated'][0, -1] or batch['truncated'][0, -1]):
        raise ValueError('Diagnostics require complete episode endpoints')
    state, _, _ = model.observe_step(model.initial(1), torch.zeros_like(batch['actions'][:, 0]),
        batch['observations'][:, 0], torch.ones(1, dtype=torch.bool), stochastic=False)
    for t in range(context):
        state, _, _ = model.observe_step(state, batch['actions'][:, t], batch['observations'][:, t+1],
            torch.zeros(1, dtype=torch.bool), stochastic=False)
    prior_state, posterior_state = state.clone(), state.clone()
    records = []
    for t in range(context, length):
        # This branch receives no future observation or teacher-forced latent.
        prior_state, prior_logits = model.prior_step(prior_state, batch['actions'][:, t], stochastic=False)
        prior_heads = model.heads(prior_state)
        posterior_state, _, posterior_logits = model.observe_step(posterior_state, batch['actions'][:, t],
            batch['observations'][:, t+1], torch.zeros(1, dtype=torch.bool), stochastic=False)
        posterior_heads = model.heads(posterior_state)
        q, p = model.probabilities(posterior_logits), model.probabilities(prior_logits)
        tiny = torch.finfo(q.dtype).tiny
        record = {'horizon': t-context+1, 'transition': t,
                  'reward': float(batch['rewards'][0, t]),
                  'continuation': float(batch['continuation'][0, t]),
                  'terminated': bool(batch['terminated'][0, t]),
                  'truncated': bool(batch['truncated'][0, t]),
                  'posterior_to_open_loop_prior_kl': float((q*(q.clamp_min(tiny).log()-p.clamp_min(tiny).log())).sum()),
                  'memory_mse': float((posterior_state.deter-prior_state.deter).square().mean())}
        for mode, heads in [('prior', prior_heads), ('posterior', posterior_heads)]:
            predicted_obs = symexp(heads['observation_symlog'])
            record[mode] = {'reward': float(symexp(heads['reward_symlog'])[0]),
                            'continuation': float(heads['continuation_logit'].sigmoid()[0]),
                            'observation_mse': float((predicted_obs-batch['observations'][:, t+1]).square().mean())}
        record['reward_head_gap'] = abs(record['prior']['reward']-record['posterior']['reward'])
        record['continuation_head_gap'] = abs(record['prior']['continuation']-record['posterior']['continuation'])
        records.append(record)
    if any(not np.isfinite(value) for record in records for value in
           [record['posterior_to_open_loop_prior_kl'], record['memory_mse'], record['reward_head_gap'], record['continuation_head_gap']]):
        raise FloatingPointError('Nonfinite diagnostic gap')
    return records


def prediction_scores(records, *, constant=None, bins=5):
    """Score actual records or a constant reward/continuation baseline.

    Baseline constants must be fit on training labels, never this scoring set.
    Probabilities are clipped only for finite BCE, not for Brier/calibration.
    """
    if not isinstance(records, list) or not 1 <= len(records) <= 4096 or type(bins) is not int or not 1 <= bins <= 20:
        raise ValueError('Invalid bounded scoring request')
    if constant is not None and (not isinstance(constant, tuple) or len(constant) != 2 or
                                not np.isfinite(constant).all() or not 0 <= constant[1] <= 1):
        raise ValueError('Finite reward / probability baseline required')
    reward = np.array([r['reward'] for r in records], np.float64)
    target = np.array([r['continuation'] for r in records], np.float64)
    if not np.isfinite(reward).all() or not np.isin(target, [0., 1.]).all():
        raise ValueError('Invalid real targets')
    result = {'count': len(records), 'death_targets': int((target == 0).sum()),
              'continuing_targets': int((target == 1).sum()),
              'exclusive_truncation_targets': sum(r['truncated'] and not r['terminated'] for r in records)}
    for mode in (['constant'] if constant is not None else ['prior', 'posterior']):
        pred_reward = np.full(len(records), constant[0]) if constant is not None else np.array([r[mode]['reward'] for r in records])
        prob = np.full(len(records), constant[1]) if constant is not None else np.array([r[mode]['continuation'] for r in records])
        if not np.isfinite(pred_reward).all() or not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
            raise ValueError('Invalid model predictions')
        safe = np.clip(prob, 1e-6, 1-1e-6)
        calibration = []
        for index in range(bins):
            selected = np.minimum((prob*bins).astype(int), bins-1) == index
            count = int(selected.sum())
            calibration.append({'lower': index/bins, 'upper': (index+1)/bins, 'count': count,
                                'mean_prediction': float(prob[selected].mean()) if count else None,
                                'observed_continuation': float(target[selected].mean()) if count else None})
        result[mode] = {'reward_mse': float(np.square(pred_reward-reward).mean()),
                        'continuation_bce': float(-(target*np.log(safe)+(1-target)*np.log(1-safe)).mean()),
                        'continuation_brier': float(np.square(prob-target).mean()),
                        'continuation_accuracy': float(((prob >= .5) == target).mean()),
                        'mean_predicted_continuation': float(prob.mean()), 'observed_continuation': float(target.mean()),
                        'calibration_bins': calibration,
                        'calibration_ece': float(sum(b['count']/len(records)*abs(b['mean_prediction']-b['observed_continuation']) for b in calibration if b['count']))}
        if constant is None:
            result[mode]['observation_mse'] = float(np.mean([r[mode]['observation_mse'] for r in records]))
    if constant is None:
        result['posterior_prior_gap'] = {key: float(np.mean([r[key] for r in records])) for key in
            ['posterior_to_open_loop_prior_kl', 'memory_mse', 'reward_head_gap', 'continuation_head_gap']}
    return result
