import unittest
import torch
from survival_agent.hybrid.control import PosteriorActorCritic
from survival_agent.hybrid.world_model import WorldModel, ModelConfig
from survival_agent.hybrid.contracts import EnvSpec, ActionSpec


class PosteriorControlTests(unittest.TestCase):
    def test_only_actor_value_receive_gradients_and_features_are_detached(self):
        model = WorldModel(EnvSpec(45, ActionSpec('discrete', count=5)), ModelConfig(deter=32, hidden=32, stoch=4, classes=4))
        agent = PosteriorActorCritic(model)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        features = torch.randn(4, 48, requires_grad=True)
        logits, value = agent.predict(features)
        (logits.square().mean()+value.square().mean()).backward()
        self.assertIsNone(features.grad)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))
        self.assertTrue(all(p.grad is not None for p in agent.trainable_parameters()))
        self.assertTrue(all(p.grad is None for p in agent.target_value.parameters()))
        for key, val in before.items():torch.testing.assert_close(val, model.state_dict()[key], rtol=0, atol=0)

    def test_target_ema_and_shape(self):
        agent = PosteriorActorCritic(WorldModel(EnvSpec(2, ActionSpec('discrete', count=2))))
        original = agent.target_value.weight.clone()
        with torch.no_grad():agent.value.weight.add_(1.)
        agent.update_target(.5)
        torch.testing.assert_close(agent.target_value.weight, original+.5)
        with self.assertRaises(ValueError):agent.update_target(1.)


if __name__ == '__main__':unittest.main()
