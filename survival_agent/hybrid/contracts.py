"""Single-agent vector/action and transition contracts; no concrete simulator import."""
from dataclasses import dataclass
from copy import deepcopy
import numpy as np


def flag(value):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError('Expected boolean episode flag')
    return bool(value)


def vector(value, width):
    value = np.asarray(value)
    if value.shape != (width,) or value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
        raise ValueError('Expected finite numeric vector of declared width')
    with np.errstate(over='ignore'):
        result = value.astype(np.float32, copy=True)
    if not np.isfinite(result).all():
        raise ValueError('Vector is not representable in float32')
    return result


@dataclass(frozen=True)
class ActionSpec:
    kind: str
    count: int = 0
    low: tuple = ()
    high: tuple = ()

    def __post_init__(self):
        if self.kind == 'discrete':
            if type(self.count) is not int or not 1 <= self.count <= 4096 or self.low or self.high:
                raise ValueError('Discrete action count must be 1..4096')
        elif self.kind == 'continuous':
            if self.count != 0 or not isinstance(self.low, tuple) or not isinstance(self.high, tuple):
                raise ValueError('Continuous bounds must be immutable tuples')
            if not 1 <= len(self.low) <= 256 or len(self.low) != len(self.high):
                raise ValueError('Continuous action width must be 1..256')
            lo, hi = vector(self.low, len(self.low)), vector(self.high, len(self.high))
            if not (lo < hi).all() or not np.isfinite(hi.astype(np.float64)-lo).all():
                raise ValueError('Finite ordered action bounds required')
            object.__setattr__(self, 'low', tuple(float(x) for x in lo))
            object.__setattr__(self, 'high', tuple(float(x) for x in hi))
        else:
            raise ValueError('Only discrete or bounded continuous actions supported')

    def validate(self, action, mask=None):
        """Validate the replay/model action: continuous coordinates are [-1,1]."""
        if self.kind == 'discrete':
            if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) or not 0 <= int(action) < self.count:
                raise ValueError('Invalid discrete action')
            mask = self.mask(mask)
            if mask is not None and not mask[int(action)]:
                raise ValueError('Root action is not legal')
            return int(action)
        if mask is not None:
            raise ValueError('Discrete masks do not apply to continuous actions')
        value = vector(action, len(self.low))
        if ((value < -1) | (value > 1)).any():
            raise ValueError('Normalized action is outside [-1,1]')
        return value

    def mask(self, mask):
        if mask is None:
            return None
        value = np.asarray(mask)
        if self.kind != 'discrete' or value.shape != (self.count,) or value.dtype != np.bool_:
            raise ValueError('Expected an exact boolean discrete mask')
        return value.copy()

    def to_environment(self, action, mask=None):
        value = self.validate(action, mask)
        if self.kind == 'discrete':
            return value
        lo, hi = np.asarray(self.low, np.float64), np.asarray(self.high, np.float64)
        return np.clip(lo + (value.astype(np.float64)+1)*0.5*(hi-lo), lo, hi).astype(np.float32)

    def from_environment(self, action):
        if self.kind == 'discrete':
            return self.validate(action)
        value = vector(action, len(self.low)).astype(np.float64)
        lo, hi = np.asarray(self.low), np.asarray(self.high)
        if ((value < lo) | (value > hi)).any():
            raise ValueError('Environment action is outside bounds')
        return self.validate((2*(value-lo)/(hi-lo)-1).astype(np.float32))

    def planner_name(self, planning=True):
        """Dispatch contract only. This does not construct an implemented planner."""
        return ('mcts' if self.kind == 'discrete' else 'mpc') if planning else 'actor'


@dataclass(frozen=True)
class EnvSpec:
    observation_width: int
    action: ActionSpec

    def __post_init__(self):
        if type(self.observation_width) is not int or not 1 <= self.observation_width <= 1024 or not isinstance(self.action, ActionSpec):
            raise ValueError('Expected bounded vector environment specification')

    def observation(self, value):
        return vector(value, self.observation_width)

    @classmethod
    def from_gymnasium(cls, env):
        # Optional import: base data contracts/replay need only NumPy.
        from gymnasium import spaces
        obs, act = env.observation_space, env.action_space
        if not isinstance(obs, spaces.Box) or len(obs.shape) != 1:
            raise ValueError('Only single-agent vector Box observations supported')
        if isinstance(act, spaces.Discrete) and act.start == 0:
            action = ActionSpec('discrete', count=int(act.n))
        elif isinstance(act, spaces.Box) and len(act.shape) == 1 and act.dtype == np.float32:
            action = ActionSpec('continuous', low=tuple(float(x) for x in act.low), high=tuple(float(x) for x in act.high))
        else:
            raise ValueError('Unsupported, mixed, nonzero-start or unbounded action space')
        return cls(int(obs.shape[0]), action)


class MemorySlots:
    """Lifecycle placeholders, not an RSSM: independent state for each stream."""
    def __init__(self):
        self.recurrent, self.warm_start = {}, {}

    def set(self, stream, recurrent, warm_start):
        if type(stream) is not int or not 0 <= stream < 1024:
            raise ValueError('Stream identifier outside bounded range')
        self.recurrent[stream] = deepcopy(recurrent)
        self.warm_start[stream] = deepcopy(warm_start)

    def boundary(self, stream, terminated, truncated):
        terminal, truncation = flag(terminated), flag(truncated)
        if terminal or truncation:
            self.recurrent.pop(stream, None)
            self.warm_start.pop(stream, None)


@dataclass
class Transition:
    observation: np.ndarray
    action: object
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool
    mask: object = None
    next_mask: object = None
    reset_observation: object = None

    @property
    def continuation(self):
        return float(not self.terminated)  # gamma is always separate.


class VectorAdapter:
    """Five-value single-agent Gymnasium adapter with an explicit autoreset mode.

    Default: explicit reset. Optional same_step requires final_observation/final_obs.
    A vectorized env or arbitrary unknown autoreset protocol is not supported.
    """
    def __init__(self, env, spec=None, *, autoreset='disabled', memory=None, stream=0):
        if autoreset not in ('disabled', 'same_step'):
            raise ValueError('Declare the autoreset convention explicitly')
        if type(stream) is not int or not 0 <= stream < 1024:
            raise ValueError('Invalid bounded stream identifier')
        self.env = env
        self.spec = spec or EnvSpec.from_gymnasium(env)
        self.autoreset = autoreset
        self.memory = memory if memory is not None else MemorySlots()
        self.stream = stream
        self.current = None
        self.current_mask = None

    def reset(self, seed=None):
        observation, info = self.env.reset(seed=seed)
        self.current = self.spec.observation(observation)
        self.current_mask = self.spec.action.mask(info.get('action_mask'))
        self.memory.boundary(self.stream, True, False)
        return self.current.copy(), deepcopy(info)

    def step(self, action):
        if self.current is None:
            raise RuntimeError('Explicit reset required before acting')
        replay_action = self.spec.action.validate(action, self.current_mask)
        actual_action = self.spec.action.to_environment(replay_action, self.current_mask)
        previous, previous_mask = self.current.copy(), self.current_mask
        # A failed/invalid external step must never permit reuse of a stale observation.
        self.current = None
        observation, reward, terminated, truncated, info = self.env.step(actual_action)
        terminated, truncated = flag(terminated), flag(truncated)
        self.memory.boundary(self.stream, terminated, truncated)
        if isinstance(reward, (bool, np.bool_)) or not isinstance(reward, (int,float,np.integer,np.floating)) or not np.isfinite(reward):
            raise ValueError('Finite scalar reward required')
        with np.errstate(over='ignore'):
            if not np.isfinite(np.float32(reward)):
                raise ValueError('Reward is not representable in float32')
        boundary = terminated or truncated
        final_key = next((key for key in ('final_observation', 'final_obs') if key in info), None)
        reset_observation = None
        next_info = info
        if boundary and self.autoreset == 'same_step':
            if final_key is None:
                raise ValueError('Same-step autoreset must provide the true final observation')
            reset_observation = self.spec.observation(observation)
            observation = info[final_key]
            next_info = info.get('final_info', {})
        elif final_key is not None:
            raise ValueError('Autoreset information requires explicit same_step configuration')
        final_observation = self.spec.observation(observation)
        next_mask = self.spec.action.mask(next_info.get('action_mask'))
        result = Transition(previous, replay_action, float(reward), final_observation.copy(),
                            terminated, truncated, previous_mask, next_mask, reset_observation)
        self.current = reset_observation if boundary else final_observation
        self.current_mask = self.spec.action.mask(info.get('action_mask')) if reset_observation is not None else next_mask
        return result


def discounted_return(rewards, terminated, truncated, final_value, gamma):
    """One contiguous sequence; terminal reward is retained, reset value is forbidden."""
    rewards = np.asarray(rewards)
    terminated, truncated = np.asarray(terminated), np.asarray(truncated)
    if rewards.ndim != 1 or not 1 <= len(rewards) <= 4096 or terminated.shape != rewards.shape or truncated.shape != rewards.shape:
        raise ValueError('Exact bounded return shapes required')
    if rewards.dtype.kind not in 'fiu' or terminated.dtype != np.bool_ or truncated.dtype != np.bool_ or not np.isfinite(rewards).all() or not np.isscalar(final_value) or not np.isfinite(final_value) or not 0 <= gamma <= 1:
        raise ValueError('Invalid return inputs')
    if (terminated[:-1] | truncated[:-1]).any():
        raise ValueError('Return cannot cross an episode boundary')
    value = float(final_value)
    for reward, terminal in zip(rewards[::-1], terminated[::-1]):
        value = float(reward) + gamma * float(not terminal) * value
    return value
