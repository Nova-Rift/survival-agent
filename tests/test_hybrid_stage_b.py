from copy import deepcopy
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from survival_agent.hybrid.contracts import EnvSpec, ActionSpec
from survival_agent.hybrid.replay import SequenceReplay
from survival_agent.hybrid import checkpoints
from survival_agent.hybrid.world_model import (WorldModel, ModelConfig, WorldModelTrainer,
    Latent, symlog, symexp, sequence_batch, sequence_loss, prediction_diagnostics)


def replay_fixture(continuous=False):
    action = ActionSpec('continuous', low=(-2.,), high=(2.,)) if continuous else ActionSpec('discrete', count=2)
    replay = SequenceReplay(EnvSpec(2, action), seed=7)
    for offset, terminal in [(0, True), (10, False)]:
        replay.start(np.array([offset, 0], np.float32))
        for t in range(4):
            replay.append(np.array([.5], np.float32) if continuous else t % 2,
                          t+1., np.array([offset+t+1, t%2], np.float32),
                          terminal and t == 3, not terminal and t == 3)
    return replay


def small_model(spec, **kwargs):
    return WorldModel(spec, ModelConfig(deter=16, hidden=16, stoch=2, classes=3, **kwargs))


class WorldModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        torch.set_num_threads(1)

    def test_shapes_discrete_continuous_stochastic_and_prior_without_decoder(self):
        for continuous in [False, True]:
            replay = replay_fixture(continuous)
            model = small_model(replay.spec)
            state = model.initial(2)
            actions = torch.zeros(2, 1) if continuous else torch.zeros(2, dtype=torch.int64)
            state, prior, post = model.observe_step(state, actions, torch.ones(2, 2), torch.ones(2, dtype=torch.bool))
            self.assertEqual(state.deter.shape, (2, 16))
            self.assertEqual(state.stoch.shape, (2, 2, 3))
            torch.testing.assert_close(state.stoch.sum(-1), torch.ones(2, 2))
            self.assertEqual(prior.shape, post.shape)
            self.assertTrue(set(state.stoch.detach().flatten().tolist()) <= {0., 1.})
            with patch.object(model.decoder, 'forward', side_effect=AssertionError('decoder must be optional')):
                future, _ = model.prior_step(state, actions)
                self.assertEqual(set(model.heads(future, decode=False)), {'reward_symlog', 'continuation_logit'})
            losses = sequence_loss(model, replay.sample(2, burn_in=1, unroll=3))
            losses['total'].backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_prior_is_causal_posterior_uses_observation_and_memory_is_recurrent(self):
        model = small_model(replay_fixture().spec)
        initial = model.initial(1)
        action = torch.tensor([1])
        first, _, _ = model.observe_step(initial, action, torch.zeros(1, 2), torch.ones(1, dtype=torch.bool), stochastic=False)
        a, pa, qa = model.observe_step(first, action, torch.zeros(1, 2), torch.zeros(1, dtype=torch.bool), stochastic=False)
        b, pb, qb = model.observe_step(first, action, torch.ones(1, 2)*10, torch.zeros(1, dtype=torch.bool), stochastic=False)
        torch.testing.assert_close(a.deter, b.deter)
        torch.testing.assert_close(pa, pb)
        self.assertFalse(torch.equal(qa, qb))
        direct, pdirect = model.prior_step(first, action, stochastic=False)
        torch.testing.assert_close(direct.deter, a.deter)
        torch.testing.assert_close(pdirect, pa)
        altered = first.clone(); altered.deter = altered.deter+1
        c, _ = model.prior_step(altered, action, stochastic=False)
        self.assertFalse(torch.equal(c.deter, direct.deter))
        torch.testing.assert_close(first.deter, first.clone().deter)

    def test_resets_both_components_and_previous_action_per_stream(self):
        model = small_model(replay_fixture().spec)
        state = Latent(torch.randn(2, 16), torch.randn(2, 2, 3))
        untouched = state.clone()
        result, _, _ = model.observe_step(state, torch.ones(2, dtype=torch.int64), torch.ones(2, 2), torch.tensor([True, False]), stochastic=False)
        fresh, _, _ = model.observe_step(model.initial(1), torch.zeros(1, dtype=torch.int64), torch.ones(1, 2), torch.tensor([True]), stochastic=False)
        other, _, _ = model.observe_step(Latent(state.deter[1:], state.stoch[1:]), torch.ones(1, dtype=torch.int64), torch.ones(1, 2), torch.tensor([False]), stochastic=False)
        torch.testing.assert_close(result.deter[:1], fresh.deter)
        torch.testing.assert_close(result.stoch[:1], fresh.stoch)
        torch.testing.assert_close(result.deter[1:], other.deter)
        torch.testing.assert_close(state.deter, untouched.deter)
        torch.testing.assert_close(state.stoch, untouched.stoch)

    def test_kl_gradient_stops_uniform_mix_and_free_nats(self):
        model = small_model(replay_fixture().spec, free_nats=0.)
        post = torch.randn(1, 2, 3, requires_grad=True)
        prior = torch.randn(1, 2, 3, requires_grad=True)
        dyn, rep = model.balanced_kl(post, prior)
        q, p = model.probabilities(post), model.probabilities(prior)
        expected = (q*(q.log()-p.log())).sum((-1, -2))
        torch.testing.assert_close(dyn, expected)
        torch.testing.assert_close(rep, expected)
        dyn.sum().backward(retain_graph=True)
        self.assertIsNone(post.grad); self.assertGreater(float(prior.grad.abs().sum()), 0)
        prior.grad = None
        rep.sum().backward()
        self.assertIsNone(prior.grad); self.assertGreater(float(post.grad.abs().sum()), 0)
        extreme = model.probabilities(torch.tensor([[[1000., -1000., -1000.]]]))
        self.assertGreaterEqual(float(extreme.min()), model.config.unimix/3-1e-9)
        floored = small_model(replay_fixture().spec, free_nats=1.)
        x = torch.zeros(1, 2, 3, requires_grad=True)
        y = torch.zeros_like(x, requires_grad=True)
        d, r = floored.balanced_kl(x, y)
        self.assertEqual(float(d), 1.); self.assertEqual(float(r), 1.)
        (d+r).sum().backward()
        self.assertEqual(float(x.grad.abs().sum()+y.grad.abs().sum()), 0.)

    def test_symlog_symexp_round_trip_signed_values(self):
        x = torch.tensor([-100., -1., 0., 1., 100.])
        torch.testing.assert_close(symexp(symlog(x)), x)

    def test_exact_next_observation_reward_and_continuation_indices(self):
        replay = replay_fixture()
        rows = []
        for episode in replay.episodes:
            row = deepcopy(episode)
            row.update(loss_mask=np.array([False, True, True, True]),
                       context_is_episode_start=True, episode_offset=0,
                       continuation=np.zeros(4, np.float32))  # must not be trusted
            rows.append(row)
        model = small_model(replay.spec)
        for p in model.parameters():
            torch.nn.init.zeros_(p)
        model.continuation[-1].bias.data.fill_(2.)
        batch = sequence_batch(rows, replay.spec)
        self.assertEqual(batch['continuation'][0, -1], 0.)
        self.assertEqual(batch['continuation'][1, -1], 1.)  # truncation bootstraps
        calls = []
        original = model.observe_step
        def observe(state, action, obs, first, **kwargs):
            calls.append((action.clone(), obs.clone(), first.clone(), torch.is_grad_enabled()))
            return original(state, action, obs, first, **kwargs)
        with patch.object(model, 'observe_step', side_effect=observe):
            losses = sequence_loss(model, rows, stochastic=False)
        self.assertEqual(len(calls), 5)
        torch.testing.assert_close(calls[0][1], batch['observations'][:, 0])
        for t in range(4):
            torch.testing.assert_close(calls[t+1][0], batch['actions'][:, t])
            torch.testing.assert_close(calls[t+1][1], batch['observations'][:, t+1])
        self.assertFalse(calls[0][3]); self.assertFalse(calls[1][3]); self.assertTrue(calls[2][3])
        expected_rec = symlog(batch['observations'][:, 2:]).square().mean()
        expected_rew = symlog(batch['rewards'][:, 1:]).square().mean()
        expected_cont = torch.nn.functional.binary_cross_entropy_with_logits(torch.full((2, 3), 2.), batch['continuation'][:, 1:])
        torch.testing.assert_close(losses['reconstruction'], expected_rec)
        torch.testing.assert_close(losses['reward'], expected_rew)
        torch.testing.assert_close(losses['continuation'], expected_cont)

    def test_burn_in_has_no_gradient_to_context_observations(self):
        replay = replay_fixture()
        rows = replay.sample(1, burn_in=2, unroll=2)
        model = small_model(replay.spec)
        batch = sequence_batch(rows, replay.spec)
        batch['observations'].requires_grad_(True)
        with patch('survival_agent.hybrid.model.sequence_batch', return_value=batch):
            sequence_loss(model, rows, stochastic=False)['total'].backward()
        grads = batch['observations'].grad
        self.assertEqual(float(grads[:, :3].abs().sum()), 0.)
        self.assertGreater(float(grads[:, 3:].abs().sum()), 0.)

    def test_diagnostics_condition_only_on_prefix_and_use_prior_for_future(self):
        replay = replay_fixture()
        rows = replay.sample(2, unroll=4)
        model = small_model(replay.spec)
        observations_seen = []
        original = model.observe_step
        def observe(state, action, obs, first, **kwargs):
            observations_seen.append(obs.clone())
            return original(state, action, obs, first, **kwargs)
        with patch.object(model, 'observe_step', side_effect=observe), patch.object(model, 'prior_step', wraps=model.prior_step) as prior:
            report = prediction_diagnostics(model, rows, context=1, horizons=(1, 3))
        self.assertEqual(len(observations_seen), 2)
        self.assertEqual(prior.call_count, 3)
        self.assertEqual(set(report), {'1', '3'})
        self.assertTrue(all(math.isfinite(v) for r in report.values() for v in r.values()))
        with self.assertRaises(ValueError):prediction_diagnostics(model, rows, context=3, horizons=(3,))

    def test_bounded_config_replay_validation_and_continuous_actions(self):
        for kw in [{'deter': 10000}, {'stoch': True}, {'classes': 1}, {'unimix': float('nan')}, {'free_nats': -1}]:
            with self.assertRaises(ValueError):ModelConfig(**kw)
        replay = replay_fixture()
        row = replay.sample(1, unroll=4)[0]
        for key, value in [('observations', np.zeros((4, 2), np.float32)), ('actions', np.zeros(4, np.float32)),
                           ('terminated', np.array([True, False, False, True])),
                           ('is_first', np.ones(5, bool)), ('loss_mask', np.array([True, False, True, True])),
                           ('context_is_episode_start', False)]:
            bad = deepcopy(row); bad[key] = value
            with self.assertRaises(ValueError):sequence_batch([bad], replay.spec)
        bad = deepcopy(row); bad['observations'][0, 0] = np.nan
        with self.assertRaises(ValueError):sequence_batch([bad], replay.spec)
        model = small_model(replay_fixture(True).spec)
        with self.assertRaises(ValueError):model.action_features(torch.tensor([[1.01]]))
        with self.assertRaises(ValueError):WorldModelTrainer(model, learning_rate=1.)

    def test_optimizer_changes_world_model_with_finite_loss_and_reduces_fixed_fit_error(self):
        replay = replay_fixture()
        rows = replay.sample(2, burn_in=1, unroll=3)
        model = small_model(replay.spec, free_nats=0.)
        trainer = WorldModelTrainer(model, learning_rate=.003)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():initial = float(sequence_loss(model, rows, stochastic=False)['total'])
        for _ in range(25):
            metrics = trainer.step(rows)
            self.assertTrue(all(math.isfinite(v) for v in metrics.values()))
        with torch.no_grad():final = float(sequence_loss(model, rows, stochastic=False)['total'])
        self.assertLess(final, initial)  # fixed tiny fixture fit, not survival improvement
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in model.state_dict().items()))

    def test_safe_checkpoint_restores_model_optimizer_replay_rng_and_next_update(self):
        replay = replay_fixture()
        model = small_model(replay.spec)
        trainer = WorldModelTrainer(model)
        trainer.step(replay.sample(2, burn_in=1, unroll=3))
        rng = np.random.default_rng(8)
        config = {'stage': 'B', 'model': asdict(model.config),
                  'model_source_sha256': hashlib.sha256(Path(__import__('survival_agent.hybrid.world_model', fromlist=['']).__file__).read_bytes()).hexdigest()}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.safe'
            checkpoints.save(path, replay, components={'model': model.state_dict(), 'optimizer': trainer.optimizer.state_dict()},
                             config=config, rng=rng, counters={'updates': trainer.updates})
            expected = trainer.step(replay.sample(2, burn_in=1, unroll=3))
            expected_weights = deepcopy(model.state_dict())
            resumed_model = small_model(replay.spec)
            resumed = WorldModelTrainer(resumed_model)
            restored, components, counters = checkpoints.load(path, replay.spec, config=config, rng=rng)
            resumed_model.load_state_dict(components['model'])
            resumed.optimizer.load_state_dict(components['optimizer'])
            resumed.updates = counters['updates']
            actual = resumed.step(restored.sample(2, burn_in=1, unroll=3))
            self.assertEqual(expected, actual)
            self.assertEqual(resumed.updates, trainer.updates)
            for key, value in expected_weights.items():
                torch.testing.assert_close(value, resumed_model.state_dict()[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
