from copy import deepcopy
from dataclasses import asdict
import random
from pathlib import Path
import tempfile
import unittest
try:
    import gymnasium as gym
except ImportError:
    gym = None
import numpy as np
import torch
from survival_agent.hybrid.contracts import ActionSpec,EnvSpec,MemorySlots,VectorAdapter,discounted_return
from survival_agent.hybrid.replay import SequenceReplay
from survival_agent.hybrid import checkpoints
from survival_agent import checkpoint_io
if gym is not None:
    from survival_agent.hybrid.toy import DelayedReward
    from examples.hybrid_stage_a import run


def replay_fixture(seed=0, continuous=False):
    action=ActionSpec('continuous',low=(-2.,),high=(2.,)) if continuous else ActionSpec('discrete',count=2)
    replay=SequenceReplay(EnvSpec(2,action),seed=seed)
    for offset,truncated in [(0,False),(10,True)]:
        replay.start(np.array([offset,0],np.float32),None if continuous else np.array([True,True]))
        for step in range(4):
            replay.append(np.array([0.5],np.float32) if continuous else step%2,float(step+1),
                          np.array([offset+step+1,step%2],np.float32),step==3 and not truncated,
                          step==3 and truncated,None if continuous else np.array([True,True]))
    return replay


class ContractTests(unittest.TestCase):
    def test_continuous_normalization_bounds_and_dispatch(self):
        action=ActionSpec('continuous',low=(-2.,0.),high=(2.,4.))
        for value in [np.array([-1,1]),np.array([0,0]),np.array([1,-1])]:
            actual=action.to_environment(value)
            np.testing.assert_allclose(action.from_environment(actual),value,atol=1e-6)
            self.assertTrue((actual>=action.low).all() and (actual<=action.high).all())
        for value in [[1.01,0],[np.nan,0],[0],['x','y']]:
            with self.assertRaises(ValueError):action.validate(value)
        with self.assertRaises(ValueError):ActionSpec('continuous',low=(0.,),high=(np.inf,))
        self.assertEqual(action.planner_name(),'mpc')
        self.assertEqual(ActionSpec('discrete',count=2).planner_name(),'mcts')
        self.assertEqual(action.planner_name(False),'actor')

    @unittest.skipIf(gym is None, 'optional Gymnasium dependency unavailable')
    def test_root_constraints_and_invalid_actions_never_step_environment(self):
        class Masked(DelayedReward):
            calls=0
            def reset(self,**kwargs):
                observation,_=super().reset(**kwargs)
                return observation,{'action_mask':np.array([True,False])}
            def step(self,action):
                self.calls+=1
                return super().step(action)
        env=Masked();adapter=VectorAdapter(env);adapter.reset(seed=0)
        for invalid in [True,2,1,0.5]:
            with self.assertRaises(ValueError):adapter.step(invalid)
        self.assertEqual(env.calls,0)

    @unittest.skipIf(gym is None, 'optional Gymnasium dependency unavailable')
    def test_real_cartpole_and_pendulum_step_contract(self):
        for name in ['CartPole-v1','Pendulum-v1']:
            with self.subTest(name=name):
                env=gym.make(name,max_episode_steps=1)
                try:
                    adapter=VectorAdapter(env);first,_=adapter.reset(seed=3)
                    adapter.memory.set(0,np.ones(2),np.ones(2))
                    action=0 if adapter.spec.action.kind=='discrete' else np.array([1.],np.float32)
                    transition=adapter.step(action)
                    np.testing.assert_array_equal(transition.observation,first)
                    self.assertTrue(transition.truncated)
                    self.assertEqual(transition.continuation,1.)
                    self.assertNotIn(0,adapter.memory.recurrent)
                    self.assertNotIn(0,adapter.memory.warm_start)
                    with self.assertRaises(RuntimeError):adapter.step(action)
                finally:env.close()

    def test_same_step_final_observation_is_not_reset_observation(self):
        class SameStep:
            def reset(self,seed=None):return np.array([0.],np.float32),{}
            def step(self,action):return np.array([-100.],np.float32),2.,False,True,{'final_observation':np.array([4.],np.float32)}
        spec=EnvSpec(1,ActionSpec('discrete',count=2))
        adapter=VectorAdapter(SameStep(),spec,autoreset='same_step')
        observation,_=adapter.reset();replay=SequenceReplay(spec);replay.start(observation)
        adapter.memory.set(0,np.array([3]),np.array([9]))
        transition=adapter.step(0)
        self.assertEqual(transition.next_observation[0],4)
        self.assertEqual(transition.reset_observation[0],-100)
        self.assertEqual(adapter.current[0],-100)
        self.assertEqual(adapter.memory.recurrent,{})
        replay.append(0,2.,transition.next_observation,False,True)
        self.assertEqual(replay.episodes[0]['observations'][-1,0],4)
        self.assertEqual(discounted_return([2.],np.array([False]),np.array([True]),transition.next_observation[0],.5),4.)
        wrong_mode=VectorAdapter(SameStep(),spec);wrong_mode.reset()
        with self.assertRaises(ValueError):wrong_mode.step(0)
        with self.assertRaises(RuntimeError):wrong_mode.step(0)

    def test_reset_lifecycle_is_per_stream_for_both_end_flags(self):
        slots=MemorySlots()
        for terminal,truncated in [(True,False),(False,True),(True,True)]:
            slots.set(0,[1],[2]);slots.set(1,[3],[4])
            slots.boundary(0,terminal,truncated)
            self.assertNotIn(0,slots.recurrent);self.assertNotIn(0,slots.warm_start)
            self.assertEqual(slots.recurrent[1],[3]);self.assertEqual(slots.warm_start[1],[4])

    def test_numeric_returns_keep_terminal_reward_and_gamma_separate(self):
        self.assertEqual(discounted_return([1.,2.],np.array([False,False]),np.array([False,True]),4.,.5),3.)
        self.assertEqual(discounted_return([1.,2.],np.array([False,True]),np.array([False,False]),4.,.5),2.)
        with self.assertRaises(ValueError):discounted_return([1.,2.],np.array([False,False]),np.array([True,False]),4.,.5)
        with self.assertRaises(ValueError):discounted_return([1.,2.],np.zeros((2,1),bool),np.zeros(2,bool),4.,.5)


class ReplayTests(unittest.TestCase):
    def test_alignment_burn_in_masks_and_no_episode_crossing(self):
        replay=replay_fixture()
        for row in replay.sample(16,burn_in=1,unroll=2):
            np.testing.assert_array_equal(row['loss_mask'],[False,True,True])
            np.testing.assert_array_equal(row['is_first'],[True,False,False,False])
            np.testing.assert_array_equal(row['rewards'],[1,2,3])
            np.testing.assert_array_equal(row['actions'],[0,1,0])
            np.testing.assert_array_equal(np.diff(row['observations'][:,0]),[1,1,1])
            self.assertTrue(row['context_is_episode_start'])
            self.assertEqual(row['masks'].shape,(4,2))
        for row in replay.sample(32,burn_in=1,unroll=1,require_episode_start=False):
            offset=row['episode_offset']
            self.assertEqual(row['is_first'][0],offset==0)
            self.assertEqual(row['context_is_episode_start'],offset==0)
        with self.assertRaises(ValueError):replay.sample(burn_in=2,unroll=3)
        with self.assertRaises(ValueError):replay.sample(batch_size=10000)

    def test_continuation_terminal_vs_truncation_and_copy_isolation(self):
        replay=replay_fixture(continuous=True)
        self.assertEqual(replay.episodes[0]['terminated'][-1],True)
        rows=replay.sample(32,unroll=4)
        self.assertEqual({float(row['continuation'][-1]) for row in rows},{0.,1.})
        rows[0]['observations'][:]=999
        self.assertFalse(any((e['observations']==999).any() for e in replay.episodes))

    def test_pending_episode_rejected_and_budget_evicts_whole_episodes(self):
        spec=EnvSpec(1,ActionSpec('discrete',count=2))
        replay=SequenceReplay(spec,capacity=2)
        for seed in range(3):
            replay.start([seed]);replay.append(0,1.,[seed+1],True,False)
        self.assertEqual(len(replay.episodes),2)
        replay.start([100])
        with self.assertRaises(RuntimeError):replay.start([200])
        with self.assertRaises(ValueError):replay.state_dict()
        with self.assertRaises(ValueError):replay.append(0,np.nan,[101],False,False)
        self.assertEqual(len(replay.pending['actions']),0)

    def test_malformed_checkpoint_shapes_and_masks_rejected(self):
        replay=replay_fixture();state=replay.state_dict()
        for key,value in [('terminated',np.zeros((4,1),bool)),('observations',np.zeros((5,3),np.float32)),('actions',np.zeros(4,np.float32)),('masks',np.zeros((5,2),bool))]:
            bad=deepcopy(state);bad['episodes'][0][key]=value
            with self.assertRaises(ValueError):SequenceReplay.from_state_dict(bad,replay.spec)


class CheckpointTests(unittest.TestCase):
    @unittest.skipIf(gym is None, 'optional Gymnasium dependency unavailable')
    def test_rng_replay_components_and_fixed_seed_evaluation_resume(self):
        replay=replay_fixture(seed=8);rng=np.random.default_rng(9)
        random.seed(10);np.random.seed(11);torch.manual_seed(12)
        weights=torch.tensor([[1.,-1.],[.5,2.]])
        config={'task':'toy','gamma':.5,'stage':'A'}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'stage.safe'
            checkpoints.save(path,replay,components={'weights':weights},config=config,rng=rng,counters={'steps':8})
            expected=(random.random(),np.random.random(),rng.random(),torch.rand(2))
            expected_rows=replay.sample(2,burn_in=1,unroll=2)
            restored,components,counters=checkpoints.load(path,replay.spec,config=config,rng=rng)
            actual=(random.random(),np.random.random(),rng.random(),torch.rand(2))
            self.assertEqual(expected[:3],actual[:3]);torch.testing.assert_close(expected[3],actual[3],rtol=0,atol=0)
            for a,b in zip(expected_rows,restored.sample(2,burn_in=1,unroll=2)):
                np.testing.assert_array_equal(a['observations'],b['observations'])
            self.assertEqual(counters,{'steps':8})
            def evaluate(matrix):
                env=DelayedReward();adapter=VectorAdapter(env);observation,_=adapter.reset(seed=7);rewards=[]
                while True:
                    action=int((matrix@torch.from_numpy(observation)).argmax())
                    transition=adapter.step(action);rewards.append(transition.reward);observation=transition.next_observation
                    if transition.terminated or transition.truncated:return rewards,observation
            first=evaluate(weights);second=evaluate(components['weights'])
            self.assertEqual(first[0],second[0]);np.testing.assert_array_equal(first[1],second[1])

    def test_schema_provenance_nonfinite_and_legacy_rejection_before_rng_mutation(self):
        replay=replay_fixture();rng=np.random.default_rng(2);config={'task':'toy'}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'stage.safe'
            checkpoints.save(path,replay,components={},config=config,rng=rng,counters={})
            rng_before=deepcopy(rng.bit_generator.state)
            with self.assertRaises(ValueError):checkpoints.load(path,replay.spec,config={'task':'different'},rng=rng)
            self.assertEqual(rng_before,rng.bit_generator.state)
            state=checkpoint_io.load(path);state['source']='wrong';checkpoint_io.save(state,path)
            with self.assertRaises(ValueError):checkpoints.load(path,replay.spec,config=config,rng=rng)
            with self.assertRaises(ValueError):checkpoints.save(path,replay,components={},config={'gamma':float('inf')},rng=rng,counters={})
            path.write_bytes(b'legacy pickle refused')
            with self.assertRaises(ValueError):checkpoints.load(path,replay.spec,config=config,rng=rng)

    @unittest.skipIf(gym is None, 'optional Gymnasium dependency unavailable')
    def test_complete_cpu_smoke_has_no_learning_claim_or_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            report=run(Path(folder)/'smoke')
            self.assertEqual(len(report['runs']),3)
            self.assertEqual(report['optimizer_updates'],0)
            self.assertFalse(report['learning_claim'])
            self.assertFalse(report['world_model_implemented'])


if __name__=='__main__':unittest.main()
