"""Bounded complete-episode sequence replay in real observation/action units.

Continuous actions use normalized coordinates. No stale latent states are stored.
"""
from copy import deepcopy
from dataclasses import asdict
import numpy as np
from .contracts import EnvSpec, ActionSpec, flag, vector


FIELDS = {'observations', 'actions', 'rewards', 'terminated', 'truncated', 'is_first', 'masks'}


class SequenceReplay:
    def __init__(self, spec, *, seed=0, capacity=4096, max_bytes=8*1024*1024):
        if not isinstance(spec, EnvSpec) or type(capacity) is not int or not 1 <= capacity <= 16384:
            raise ValueError('Invalid bounded replay capacity')
        if type(max_bytes) is not int or not 1024 <= max_bytes <= 16*1024*1024:
            raise ValueError('Invalid bounded replay byte budget')
        self.spec, self.capacity, self.max_bytes = spec, capacity, max_bytes
        self.rng = np.random.default_rng(seed)
        self.episodes = []
        self.pending = None

    def start(self, observation, mask=None):
        if self.pending is not None:
            raise RuntimeError('Cannot reset an unfinished replay episode')
        observation = self.spec.observation(observation)
        mask = self.spec.action.mask(mask)
        self.pending = {'observations': [observation], 'actions': [], 'rewards': [],
                        'terminated': [], 'truncated': [], 'masks': [mask]}

    def append(self, action, reward, next_observation, terminated, truncated, next_mask=None):
        if self.pending is None:
            raise RuntimeError('Start an episode before appending')
        episode = self.pending
        terminal, truncation = flag(terminated), flag(truncated)
        observation = self.spec.observation(next_observation)
        mask = self.spec.action.mask(next_mask)
        if (mask is None) != (episode['masks'][0] is None):
            raise ValueError('Mask availability must be consistent within an episode')
        action = self.spec.action.validate(action, episode['masks'][-1])
        if isinstance(reward, (bool, np.bool_)) or not isinstance(reward, (int, float, np.integer, np.floating)) or not np.isfinite(reward):
            raise ValueError('Finite scalar reward required')
        with np.errstate(over='ignore'):
            reward32 = np.float32(reward)
        if not np.isfinite(reward32):
            raise ValueError('Reward not representable in float32')
        n = len(episode['actions']) + 1
        action_width = 1 if self.spec.action.kind == 'discrete' else len(self.spec.action.low)
        estimate = (n+1)*self.spec.observation_width*4 + n*(action_width*8+6) + n+1
        if mask is not None:
            estimate += (n+1)*self.spec.action.count
        if n > min(4096, self.capacity) or estimate > self.max_bytes:
            raise ValueError('Episode exceeds replay bounds; collector must use a shorter horizon')
        episode['actions'].append(action)
        episode['rewards'].append(float(reward32))
        episode['terminated'].append(terminal)
        episode['truncated'].append(truncation)
        episode['observations'].append(observation)
        episode['masks'].append(mask)
        if terminal or truncation:
            packed = {key: np.asarray(episode[key], dtype=np.float32 if key in ['observations','rewards'] else
                                      np.int64 if key == 'actions' and self.spec.action.kind == 'discrete' else
                                      np.float32 if key == 'actions' else bool)
                      for key in ['observations','actions','rewards','terminated','truncated']}
            packed['is_first'] = np.zeros(n+1, bool)
            packed['is_first'][0] = True
            packed['masks'] = None if mask is None else np.asarray(episode['masks'], bool)
            self.validate_episode(packed)
            self.episodes.append(packed)
            self.pending = None
            while sum(len(e['actions']) for e in self.episodes) > self.capacity or sum(self.nbytes(e) for e in self.episodes) > self.max_bytes:
                self.episodes.pop(0)

    @staticmethod
    def nbytes(episode):
        return sum(value.nbytes for value in episode.values() if isinstance(value, np.ndarray))

    def validate_episode(self, episode):
        if not isinstance(episode, dict) or set(episode) != FIELDS:
            raise ValueError('Invalid replay episode schema')
        if any(not isinstance(episode[key], np.ndarray) for key in FIELDS - {'masks'}):
            raise ValueError('Replay fields must be arrays')
        actions = episode['actions']
        n = actions.shape[0] if actions.ndim else 0
        if not 1 <= n <= min(4096, self.capacity) or self.nbytes(episode) > self.max_bytes:
            raise ValueError('Oversize replay episode')
        width = self.spec.observation_width
        if episode['observations'].shape != (n+1, width) or episode['observations'].dtype != np.float32:
            raise ValueError('Invalid observation sequence')
        for key in ['terminated', 'truncated']:
            if episode[key].shape != (n,) or episode[key].dtype != np.bool_:
                raise ValueError('Exact boolean transition shape required')
        if episode['rewards'].shape != (n,) or episode['rewards'].dtype != np.float32:
            raise ValueError('Exact float reward shape required')
        if episode['is_first'].shape != (n+1,) or episode['is_first'].dtype != np.bool_ or not episode['is_first'][0] or episode['is_first'][1:].any():
            raise ValueError('Exactly one episode start required')
        boundary = episode['terminated'] | episode['truncated']
        if not boundary[-1] or boundary[:-1].any():
            raise ValueError('Complete episodes cannot cross boundaries')
        expected = (n,) if self.spec.action.kind == 'discrete' else (n,len(self.spec.action.low))
        dtype = np.int64 if self.spec.action.kind == 'discrete' else np.float32
        if actions.shape != expected or actions.dtype != dtype:
            raise ValueError('Invalid canonical action sequence')
        for value in [episode['observations'], episode['rewards'], actions]:
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite replay value')
        masks = episode['masks']
        if masks is not None:
            if self.spec.action.kind != 'discrete' or not isinstance(masks, np.ndarray) or masks.shape != (n+1,self.spec.action.count) or masks.dtype != np.bool_:
                raise ValueError('Exact boolean mask sequence required')
        for index, action in enumerate(actions):
            self.spec.action.validate(action, None if masks is None else masks[index])

    def sample(self, batch_size=1, *, burn_in=0, unroll=1, require_episode_start=True):
        if any(type(x) is not int for x in [batch_size,burn_in,unroll]) or not 1 <= batch_size <= 32 or not 0 <= burn_in <= 128 or not 1 <= unroll <= 256:
            raise ValueError('Invalid bounded sequence request')
        if type(require_episode_start) is not bool:
            raise ValueError('Declare reconstruction context explicitly')
        length = burn_in + unroll
        candidates = [(e, 1 if require_episode_start else len(e['actions'])-length+1)
                      for e in self.episodes if len(e['actions']) >= length]
        if not candidates:
            raise ValueError('No complete episode long enough; sequences are never joined or padded')
        total = sum(count for _,count in candidates)
        rows = []
        for _ in range(batch_size):
            choice = int(self.rng.integers(total))
            for episode, count in candidates:
                if choice < count:
                    break
                choice -= count
            start = 0 if require_episode_start else choice
            end = start + length
            row = {key: None if value is None else value[start:end+(key in ['observations','is_first','masks'])].copy()
                   for key,value in episode.items()}
            row['loss_mask'] = np.arange(length) >= burn_in
            row['continuation'] = (~row['terminated']).astype(np.float32)
            row['context_is_episode_start'] = start == 0
            row['episode_offset'] = start
            rows.append(row)
        return rows

    def state_dict(self):
        if self.pending is not None:
            raise ValueError('Stage-A checkpointing is at episode boundaries only')
        return deepcopy({'version':1, 'spec':asdict(self.spec), 'capacity':self.capacity,
                         'max_bytes':self.max_bytes, 'rng':self.rng.bit_generator.state, 'episodes':self.episodes})

    @classmethod
    def from_state_dict(cls, state, expected_spec):
        if not isinstance(state, dict) or set(state) != {'version','spec','capacity','max_bytes','rng','episodes'} or state['version'] != 1 or state['spec'] != asdict(expected_spec):
            raise ValueError('Replay version/spec mismatch')
        replay = cls(expected_spec, capacity=state['capacity'], max_bytes=state['max_bytes'])
        episodes = state['episodes']
        if not isinstance(episodes, list) or len(episodes) > replay.capacity:
            raise ValueError('Invalid episode collection')
        # Check totals before copying, and validate shapes before boolean arithmetic.
        for episode in episodes:
            replay.validate_episode(episode)
        if sum(len(e['actions']) for e in episodes) > replay.capacity or sum(replay.nbytes(e) for e in episodes) > replay.max_bytes:
            raise ValueError('Replay exceeds total bounds')
        replay.episodes = deepcopy(episodes)
        replay.rng.bit_generator.state = deepcopy(state['rng'])
        return replay
