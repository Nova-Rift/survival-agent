"""End-to-end collection/update/resume checks for the single-file training loop."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from survival_agent.hybrid.model import (
    TrainingConfig, TrainingSession, TransformerConfig, HybridConfig,
    LearningConfig, PlannerConfig, EnvSpec, ActionSpec,
)


class TinyEnvironment:
    """Test-only actual environment; planners never receive this object."""
    def __init__(self, continuous=False):
        self.continuous=continuous
        self.actions=[]

    def reset(self,seed=None):
        self.step_count=0
        return np.zeros(2,np.float32),{}

    def step(self,action):
        self.actions.append(deepcopy(action))
        self.step_count+=1
        self.assert_action(action)
        reward=1-float(np.asarray(action).reshape(-1)[0])**2 if self.continuous else float(action)
        return np.array([self.step_count,reward],np.float32),reward,False,False,{}

    def assert_action(self,action):
        if self.continuous:
            assert np.asarray(action).shape==(1,) and (np.abs(action)<=2).all()
        else:
            assert action in (0,1)


def session(continuous):
    spec=EnvSpec(2,ActionSpec('continuous',low=(-2.,),high=(2.,)) if continuous else ActionSpec('discrete',count=2))
    return TrainingSession(spec,TrainingConfig(seed=87201,cycles=2,initial_episodes=1,
        max_episode_steps=3,updates_per_cycle=1,batch_size=2,max_prefix=3),
        TransformerConfig(deter=16,hidden=16,stoch=2,classes=3,context=4),
        HybridConfig(hidden=16,q_ensemble=3),LearningConfig(horizon=2,distillation_weight=.1),
        PlannerConfig(horizon=2,simulations=3,candidates=8,actor_candidates=2,elites=2,iterations=1))


class OneFileTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(87201)

    def test_full_loop_resume_matches_uninterrupted_for_both_action_types(self):
        for continuous in (False,True):
            with self.subTest(continuous=continuous),tempfile.TemporaryDirectory() as directory:
                live=session(continuous);env=TinyEnvironment(continuous)
                first=live.run(env,output=Path(directory)/'first',max_cycles=1)
                self.assertEqual(first['transitions'],6)
                self.assertFalse(live.controller.states)
                self.assertFalse(live.controller.warm_starts)
                # Our collector's time limit is truncation, preserving bootstrap.
                self.assertTrue(all(e['truncated'][-1] and not e['terminated'][-1] for e in live.replay.episodes))
                expected=live.run(env)
                weights=deepcopy(live.model.state_dict())
                restored=session(continuous)
                restored.restore(Path(directory)/'first'/'checkpoint.safe')
                actual=restored.run(TinyEnvironment(continuous))
                self.assertEqual(actual,expected)
                self.assertEqual(actual['cycles'],2)
                self.assertEqual(actual['transitions'],9)
                for name,value in weights.items():
                    torch.testing.assert_close(value,restored.model.state_dict()[name],rtol=0,atol=0)
                self.assertIn('td',actual['logs'][0]['updates'][0]) if continuous else self.assertNotIn('td',actual['logs'][0]['updates'][0])

    def test_compatibility_imports_share_one_implementation(self):
        from survival_agent.hybrid import model,planning,learning,runtime,replay,world_model,policy
        self.assertIs(planning.mcts,model.mcts)
        self.assertIs(planning.mpc,model.mpc)
        self.assertIs(learning.HybridLearner,model.HybridLearner)
        self.assertIs(runtime.HybridController,model.HybridController)
        self.assertIs(replay.SequenceReplay,model.SequenceReplay)
        self.assertIs(world_model.WorldModelTrainer,model.WorldModelTrainer)
        self.assertIs(policy.sample_action,model.sample_action)


if __name__=='__main__':unittest.main()
