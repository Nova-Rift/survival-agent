from copy import deepcopy
from dataclasses import replace, asdict
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from survival_agent.hybrid.model import HybridModel, HybridConfig, TransformerConfig, TransformerLatent
from survival_agent.hybrid.contracts import EnvSpec, ActionSpec
from survival_agent.hybrid.learning import HybridLearner, LearningConfig, lambda_returns, replay_states, planner_target
from survival_agent.hybrid.planning import PlannerConfig, mcts, mpc
from survival_agent.hybrid.runtime import HybridController
from survival_agent.hybrid import checkpoints
from tests.test_hybrid_stage_b import replay_fixture


def model_for(continuous=False):
    return HybridModel(replay_fixture(continuous).spec,
        TransformerConfig(deter=16,hidden=16,stoch=2,classes=3,context=4),HybridConfig(hidden=16,q_ensemble=3))


class ToyWorld(nn.Module):
    """Known dynamics solely for planner acceptance, with no simulator methods."""
    def __init__(self, continuous):
        super().__init__()
        self.continuous = continuous

    def features(self, state):
        return state.deter

    def prior_step(self,state,action,stochastic=True):
        assert not stochastic
        step = state.deter[:,0]+1
        choice = action[:,0] if self.continuous else action.float()
        previous = state.deter[:,1]
        reward = 1-(choice-.7).square() if self.continuous else torch.where(step==1,1-choice,4*previous)
        return TransformerLatent(torch.stack([step,choice,reward],-1),state.stoch.clone(),state.history.clone(),state.lengths.clone()),None

    def heads(self,state,decode=True):
        assert not decode
        reward = state.deter[:,2]
        continuation = torch.where(state.deter[:,0]>=2,-torch.inf,torch.inf)
        return {'reward_symlog':reward.sign()*reward.abs().log1p(),'continuation_logit':continuation}


class ToyModel(nn.Module):
    def __init__(self,continuous=False):
        super().__init__()
        self.spec = replay_fixture(continuous).spec
        self.action_width = 1 if continuous else 2
        self.world_model = ToyWorld(continuous)
        self.value = nn.Linear(3,1)
        nn.init.zeros_(self.value.weight); nn.init.zeros_(self.value.bias)
        self.q_ensemble = nn.ModuleList([nn.Linear(4,1),nn.Linear(4,1)])

    def policy(self,features):
        if self.spec.action.kind == 'continuous':
            return {'mean':torch.zeros(features.shape[0],1),'std':torch.ones(features.shape[0],1)*.5}
        return {'logits':torch.zeros(features.shape[0],2)}

    def q_values(self,features,actions):
        return torch.zeros(2,features.shape[0])

    def initial(self):
        return TransformerLatent(torch.zeros(1,3),torch.zeros(1,1,1),torch.zeros(1,1,1),torch.zeros(1,dtype=torch.long))


class HybridCombinationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(731)

    def test_numeric_returns_keep_terminal_reward_and_truncation_bootstrap(self):
        r=torch.tensor([[1.,2.]])
        values=torch.tensor([[0.,0.,4.]])
        self.assertEqual(float(lambda_returns(r,torch.ones_like(r),values,.5,1)[0,0]),3.)
        self.assertEqual(float(lambda_returns(r,torch.tensor([[1.,0.]]),values,.5,1)[0,0]),2.)

    def test_imagination_updates_policy_not_world_for_both_action_types(self):
        for continuous in (False,True):
            model=model_for(continuous); learner=HybridLearner(model,LearningConfig(horizon=3))
            rows=replay_fixture(continuous).sample(2,burn_in=1,unroll=2)
            before=deepcopy(model.world_model.state_dict()); actor=deepcopy(model.actor.state_dict())
            with patch.object(model.world_model.decoder,'forward',side_effect=AssertionError('No decoding in imagination')):
                metrics=learner.imagination_step(rows)
            self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
            for key,value in before.items(): torch.testing.assert_close(value,model.world_model.state_dict()[key],rtol=0,atol=0)
            self.assertTrue(all(p.grad is None for p in model.world_model.parameters()))
            self.assertTrue(any(not torch.equal(v,model.actor.state_dict()[k]) for k,v in actor.items()))

    def test_td_targets_use_real_final_observation_and_terminal_flags(self):
        model=model_for(True); learner=HybridLearner(model,LearningConfig(gamma=.5))
        replay=replay_fixture(True)
        rows=replay.sample(2,burn_in=0,unroll=4)
        # Force one terminal and one truncated row with identical last reward.
        rows[0]['terminated'][-1]=True; rows[0]['truncated'][-1]=False
        rows[1]['terminated'][-1]=False; rows[1]['truncated'][-1]=True
        for head in learner.target_q:
            for p in head.parameters():p.data.zero_()
            head[-1].bias.data.fill_(4)
        before=deepcopy(model.world_model.state_dict())
        with patch.object(model.world_model,'prior_step',side_effect=AssertionError('TD must use real observations')):
            metrics=learner.td_step(rows)
        self.assertEqual(metrics['td_target'],[4.,6.])
        for k,v in before.items():torch.testing.assert_close(v,model.world_model.state_dict()[k],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in model.world_model.parameters()))

    def test_mcts_delayed_reward_masks_visits_and_no_live_mutation(self):
        model=ToyModel(); state=model.initial(); before=state.clone()
        config=PlannerConfig(gamma=.5,horizon=2,simulations=100,root_noise=0)
        result=mcts(model,state,config)
        self.assertEqual(result.action.item(),1)  # delayed 4 discounted to 2 beats immediate 1
        self.assertEqual(sum(result.diagnostics['visits']),100)
        masked=mcts(model,state,config,mask=torch.tensor([True,False]))
        self.assertEqual(masked.target.tolist(),[[1.,0.]])
        # Root-only legality: action 1 remains available after masked root action 0.
        self.assertGreater(masked.diagnostics['model_calls'],2)
        torch.testing.assert_close(state.deter,before.deter)
        self.assertLessEqual(result.diagnostics['model_calls'],100)
        self.assertFalse(result.target.requires_grad)
        with self.assertRaises(ValueError):mcts(model,state,config,mask=torch.tensor([False,False]))

    def test_mpc_finds_bounded_optimum_and_keeps_input_plan_and_state(self):
        model=ToyModel(True); state=model.initial(); before=state.clone()
        config=PlannerConfig(horizon=2,iterations=5,terminal_value='q')
        warm=torch.zeros(2,1); warm_copy=warm.clone()
        result=mpc(model,state,config,warm_start=warm)
        self.assertLess(abs(result.action.item()-.7),.2)
        self.assertTrue((result.action.abs()<=1).all())
        torch.testing.assert_close(warm,warm_copy)
        torch.testing.assert_close(state.deter,before.deter)
        self.assertFalse(result.target.requires_grad)
        self.assertEqual(result.warm_start.shape,(2,1))

    def test_real_neural_model_planners_and_controller_reset_dispatch(self):
        config=PlannerConfig(horizon=2,simulations=3,iterations=1,candidates=8,actor_candidates=2,elites=2)
        for continuous in (False,True):
            model=model_for(continuous)
            controller=HybridController(model,config=config)
            before=deepcopy(model.state_dict())
            with patch.object(model.world_model.decoder,'forward',side_effect=AssertionError('No search decoding')):
                plan=controller.decide(np.zeros(2,np.float32))
            self.assertEqual(plan.diagnostics['planner'],'mpc' if continuous else 'mcts')
            for _ in range(4):
                controller.decide(np.zeros(2,np.float32))
            for k,v in before.items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
            controller.boundary(terminated=False,truncated=True)
            self.assertFalse(controller.states); self.assertFalse(controller.warm_starts)
            actor=HybridController(model,planning=False)
            self.assertEqual(actor.decide(np.zeros(2,np.float32)).diagnostics['planner'],'actor')

    def test_optional_distillation_rejects_stale_and_unrelated_history(self):
        model=model_for(); config=LearningConfig(horizon=2,distillation_weight=.2)
        learner=HybridLearner(model,config)
        rows=replay_fixture().sample(1,burn_in=1,unroll=2)
        state,_,_=replay_states(model,rows)
        plan=mcts(model,state,PlannerConfig(simulations=3,horizon=2))
        teacher=planner_target(rows,plan,version=0,step=0,settings={})
        learner.imagination_step(rows,teacher)
        stale=deepcopy(teacher);stale['version']=-1
        with self.assertRaises(ValueError):learner.imagination_step(rows,stale)
        changed=deepcopy(rows);changed[0]['observations'][0,0]+=1
        with self.assertRaises(ValueError):learner.imagination_step(changed,teacher)

    def test_safe_checkpoint_resumes_optimizer_rng_and_planner(self):
        model=model_for(True);learner=HybridLearner(model,LearningConfig(horizon=2))
        replay=replay_fixture(True);rows=replay.sample(2,burn_in=1,unroll=2)
        learner.world_step(rows);learner.imagination_step(rows);learner.td_step(rows)
        config={'transformer':asdict(model.world_model.config),'hybrid':asdict(model.config),
                'learning':asdict(learner.config)}
        rng=np.random.default_rng(71)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'hybrid.safe'
            checkpoints.save(path,replay,components=learner.state_dict(),config=config,rng=rng,counters={'updates':learner.updates})
            expected=learner.imagination_step(rows)
            weights=deepcopy(model.state_dict())
            _,saved,_=checkpoints.load(path,model.spec,config=config,rng=rng)
            # Construction consumes randomness; load RNG again after construction.
            restored=HybridLearner(model_for(True),learner.config);restored.load_state_dict(saved)
            checkpoints.load(path,model.spec,config=config,rng=rng)
            actual=restored.imagination_step(rows)
            self.assertEqual(expected,actual)
            for k,v in weights.items():torch.testing.assert_close(v,restored.model.state_dict()[k],rtol=0,atol=0)
            state,_,_=replay_states(model,rows[:1])
            cfg=PlannerConfig(horizon=2,iterations=1,candidates=8,actor_candidates=2,elites=2)
            torch.manual_seed(91);a=mpc(model,state,cfg)
            torch.manual_seed(91);b=mpc(restored.model,state,cfg)
            torch.testing.assert_close(a.action,b.action,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
