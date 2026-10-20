"""Per-stream inference state and planner dispatch; only collectors execute actions."""
import torch
from .planning import PlannerConfig, Plan, mcts, mpc
from .policy import sample_action


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
