from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from survival_agent.hybrid.model import TransformerModel
from survival_agent.hybrid import checkpoints
from survival_agent.hybrid.transformer_agent import (
    TransformerConfig, TransformerLatent, TransformerWorldModel, TransformerAgent)
from survival_agent.hybrid.world_model import WorldModelTrainer, sequence_loss, sequence_batch
from tests.test_hybrid_stage_b import replay_fixture
from unittest.mock import patch


class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)

    def model(self, continuous=False, **kwargs):
        return TransformerWorldModel(replay_fixture(continuous).spec,
            TransformerConfig(deter=16, hidden=16, stoch=2, classes=3, context=3, **kwargs))

    def test_standalone_model_preserves_gradients_and_has_no_agent_policy(self):
        world = self.model()
        model = TransformerModel(world)
        self.assertTrue(world.training)
        self.assertTrue(all(p.requires_grad for p in world.parameters()))
        self.assertFalse(hasattr(model, 'act'))
        self.assertFalse(hasattr(model, 'target_value'))
        state, _, _ = world.observe_step(world.initial(1), torch.tensor([0]),
            torch.ones(1, 2), torch.tensor([True]), stochastic=False)
        logits, value = model(world.features(state))
        (logits.square().sum() + value.square().sum()).backward()
        self.assertGreater(float(world.encoder[0].weight.grad.abs().sum()), 0.)
        agent = TransformerAgent(self.model())
        model.load_state_dict({k: v for k, v in agent.state_dict().items()
                              if not k.startswith('target_value.')}, strict=True)
        features = torch.randn(2, model.actor.in_features)
        for actual, expected in zip(model(features), agent.predict(features)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_attention_is_causal_in_training_and_eval_and_padding_is_ignored(self):
        model = self.model(layers=2)
        tokens = torch.randn(2, 3, model.token_width)
        changed = tokens.clone(); changed[:, 2] += 100
        for training in (True, False):
            model.train(training)
            with torch.no_grad():
                original = model.core(tokens, torch.tensor([3, 2]))
                other = model.core(changed, torch.tensor([3, 2]))
            self.assertTrue(torch.isfinite(original).all())
            torch.testing.assert_close(original[:, :2], other[:, :2], rtol=0, atol=0)
        self.assertFalse(any(isinstance(m, torch.nn.GRUCell) for m in model.modules()))

    def test_prior_cannot_see_current_observation_and_does_not_need_decoder(self):
        model = self.model()
        state = model.initial(1); action = torch.tensor([1]); first = torch.tensor([True])
        a, pa, qa = model.observe_step(state, action, torch.zeros(1, 2), first, stochastic=False)
        b, pb, qb = model.observe_step(state, action, torch.full((1, 2), 10.), first, stochastic=False)
        torch.testing.assert_close(pa, pb, rtol=0, atol=0)
        self.assertFalse(torch.equal(qa, qb))
        with patch.object(model.decoder, 'forward', side_effect=AssertionError('No decoder in prior')):
            predicted, prior = model.prior_step(a, action, stochastic=False)
            observed, expected, _ = model.observe_step(a, action, torch.ones(1, 2), ~first, stochastic=False)
        torch.testing.assert_close(predicted.deter, observed.deter)
        torch.testing.assert_close(prior, expected)

    def test_history_rolls_and_mixed_resets_clear_only_the_correct_stream(self):
        model = self.model().eval(); state = model.initial(2)
        for t in range(7):
            old = state.clone()
            state, _, _ = model.observe_step(state, torch.tensor([t % 2, (t+1) % 2]),
                torch.tensor([[float(t), 0.], [0., float(t)]]), torch.tensor([t == 0]*2), stochastic=False)
            self.assertEqual(state.lengths.tolist(), [min(t+1, 3)]*2)
            if t >= 3:
                torch.testing.assert_close(state.history[:, :-1], old.history[:, 1:])
        untouched = state.clone()
        result, _, _ = model.observe_step(state, torch.tensor([1, 0]), torch.ones(2, 2),
                                          torch.tensor([True, False]), stochastic=False)
        fresh, _, _ = model.observe_step(model.initial(1), torch.tensor([0]), torch.ones(1, 2),
                                         torch.tensor([True]), stochastic=False)
        isolated = TransformerLatent(*(getattr(state, k)[1:] for k in ('deter', 'stoch', 'history', 'lengths')))
        other, _, _ = model.observe_step(isolated, torch.tensor([0]), torch.ones(1, 2),
                                         torch.tensor([False]), stochastic=False)
        torch.testing.assert_close(result.deter[:1], fresh.deter)
        torch.testing.assert_close(result.stoch[:1], fresh.stoch)
        torch.testing.assert_close(result.deter[1:], other.deter)
        self.assertEqual(result.lengths.tolist(), [1, 3])
        torch.testing.assert_close(result.history[0], torch.zeros_like(result.history[0]))
        torch.testing.assert_close(state.history, untouched.history, rtol=0, atol=0)
        copy = state.clone(); copy.history.zero_()
        torch.testing.assert_close(state.history, untouched.history, rtol=0, atol=0)
        self.assertFalse(state.detach().history.requires_grad)

    def test_real_gradients_and_fixed_fixture_fit_with_discrete_and_continuous_actions(self):
        for continuous in (False, True):
            replay = replay_fixture(continuous)
            model = self.model(continuous, free_nats=0.)
            rows = replay.sample(2, burn_in=1, unroll=3)
            trainer = WorldModelTrainer(model, learning_rate=.003)
            initial = float(sequence_loss(model, rows, stochastic=False)['total'])
            for _ in range(20):
                metrics = trainer.step(rows)
                self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
            for name, p in model.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(float(model.core.transformer.layers[0].self_attn.in_proj_weight.grad.abs().sum()), 0.)
            self.assertLess(float(sequence_loss(model, rows, stochastic=False)['total']), initial)

    def test_burn_in_detaches_history_and_actor_updates_cannot_change_world(self):
        replay = replay_fixture(); rows = replay.sample(1, burn_in=2, unroll=2)
        model = self.model(); batch = sequence_batch(rows, replay.spec)
        batch['observations'].requires_grad_(True)
        with patch('survival_agent.hybrid.model.sequence_batch', return_value=batch):
            sequence_loss(model, rows, stochastic=False)['total'].backward()
        self.assertEqual(float(batch['observations'].grad[:, :3].abs().sum()), 0.)
        self.assertGreater(float(batch['observations'].grad[:, 3:].abs().sum()), 0.)
        model.zero_grad(set_to_none=True)
        agent = TransformerAgent(model)
        frozen = deepcopy(model.state_dict())
        logits, value = agent.predict(torch.randn(2, agent.actor.in_features, requires_grad=True))
        optimizer = torch.optim.Adam(agent.trainable_parameters(), lr=.001)
        (logits.square().mean()+value.square().mean()).backward(); optimizer.step()
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, frozen[k], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))

    def test_safe_checkpoint_restores_next_training_update_exactly(self):
        replay = replay_fixture(); model = self.model(); trainer = WorldModelTrainer(model)
        trainer.step(replay.sample(2, burn_in=1, unroll=3))
        rng = np.random.default_rng(8)
        config = {'model': asdict(model.config)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'transformer.safe'
            checkpoints.save(path, replay, components={'model': model.state_dict(), 'optimizer': trainer.optimizer.state_dict()},
                             config=config, rng=rng, counters={'updates': 1})
            expected = trainer.step(replay.sample(2, burn_in=1, unroll=3))
            weights = deepcopy(model.state_dict())
            restored_model = self.model(); resumed = WorldModelTrainer(restored_model)
            restored, components, _ = checkpoints.load(path, replay.spec, config=config, rng=rng)
            restored_model.load_state_dict(components['model']); resumed.optimizer.load_state_dict(components['optimizer'])
            actual = resumed.step(restored.sample(2, burn_in=1, unroll=3))
            self.assertEqual(expected, actual)
            for key, value in weights.items():
                torch.testing.assert_close(value, restored_model.state_dict()[key], rtol=0, atol=0)
            with self.assertRaises(ValueError):
                checkpoints.load(path, replay.spec, config={'model': {**asdict(model.config), 'context': 4}}, rng=rng)

    def test_invalid_config_and_history_are_rejected(self):
        for config in ({'heads': 3}, {'context': 0}, {'layers': True}, {'architecture': 'gru'}, {'unimix': float('nan')}):
            with self.assertRaises(ValueError):TransformerConfig(**config)
        model = self.model(); state = model.initial(1)
        state.lengths[0] = 4
        with self.assertRaises(ValueError):model.prior_step(state, torch.tensor([0]))


if __name__ == '__main__':
    unittest.main()
