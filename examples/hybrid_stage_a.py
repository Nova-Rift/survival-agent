"""Bounded CPU data/checkpoint smoke only: no training, RSSM or planner."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import gymnasium as gym
import numpy as np
from survival_agent.hybrid.contracts import VectorAdapter
from survival_agent.hybrid.replay import SequenceReplay
from survival_agent.hybrid import checkpoints
from survival_agent.hybrid.toy import DelayedReward


def run(output, seed=0):
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a new output directory')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Seed must be a bounded nonnegative integer')
    output.mkdir(parents=True)
    reports=[]
    for name in ['CartPole-v1','Pendulum-v1','DelayedReward']:
        env = DelayedReward() if name == 'DelayedReward' else gym.make(name, max_episode_steps=3)
        try:
            adapter = VectorAdapter(env)
            rng = np.random.default_rng(seed)
            replay = SequenceReplay(adapter.spec, seed=seed)
            observation,_ = adapter.reset(seed=seed)
            replay.start(observation)
            total_reward = 0.
            steps = 0
            while True:
                action = 1 if adapter.spec.action.kind == 'discrete' else np.zeros(len(adapter.spec.action.low),np.float32)
                transition = adapter.step(action)
                replay.append(transition.action, transition.reward, transition.next_observation,
                              transition.terminated, transition.truncated)
                total_reward += transition.reward
                steps += 1
                if transition.terminated or transition.truncated:
                    break
                if steps >= 3:
                    raise AssertionError('Smoke exceeded joint-step budget')
            config = {'environment':name, 'seed':seed, 'stage':'A', 'gamma':0.5}
            path = output/(name+'.safe')
            checkpoints.save(path, replay, components={}, config=config, rng=rng, counters={'real_steps':steps})
            expected = replay.sample(unroll=2)[0]
            restored,components,counters = checkpoints.load(path, adapter.spec, config=config, rng=rng)
            actual = restored.sample(unroll=2)[0]
            np.testing.assert_array_equal(expected['observations'],actual['observations'])
            assert counters['real_steps']==steps and components=={}
            reports.append({'environment':name,'spec':asdict(adapter.spec),'steps':steps,
                            'observed_reward':total_reward,'terminated':transition.terminated,
                            'truncated':transition.truncated,'checkpoint_resume':'passed'})
        finally:
            env.close()
    report={'stage':'A data/checkpoint smoke only','runs':reports,'optimizer_updates':0,
            'planners_implemented':False,'world_model_implemented':False,'learning_claim':False,'cost_usd':0}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    print(json.dumps(run(args.output,args.seed),indent=2))


if __name__=='__main__':
    main()
