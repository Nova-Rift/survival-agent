"""Episode-boundary hybrid state, using the existing safe serializer, never pickle.

No environment or model class is dynamically reconstructed from a checkpoint.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from survival_agent import checkpoint_io
from .replay import SequenceReplay


def source_fingerprint():
    paths = [Path(__file__).with_name(name) for name in ['contracts.py','replay.py','model.py','checkpoints.py']]
    paths.append(Path(checkpoint_io.__file__))
    return hashlib.sha256(b''.join(path.read_bytes() for path in paths)).hexdigest()


def config_digest(config):
    raw = json.dumps(config, sort_keys=True, allow_nan=False, separators=(',',':')).encode()
    if len(raw) > 64*1024:
        raise ValueError('Configuration exceeds size budget')
    return hashlib.sha256(raw).hexdigest()


def versions():
    import platform
    return {'python':platform.python_version(),'numpy':np.__version__, 'torch':str(torch.__version__)}


def save(path, replay, *, components, config, rng, counters):
    if not isinstance(rng,np.random.Generator) or type(rng.bit_generator) is not np.random.PCG64:
        raise ValueError('Stage A explicitly supports PCG64 sampler state only')
    if not isinstance(components, dict) or not isinstance(counters, dict) or any(type(v) is not int or v < 0 or v > 2**63-1 for v in counters.values()):
        raise ValueError('Explicit structured components and nonnegative counters required')
    state = {'format':'hybrid-stage-a', 'version':1, 'spec':asdict(replay.spec),
             'source':source_fingerprint(), 'config_digest':config_digest(config),
             'config':config, 'versions':versions(), 'components':components,
             'replay':replay.state_dict(), 'rng':{'python':random.getstate(),
             'numpy_global':np.random.get_state(), 'sampler':rng.bit_generator.state,
             'torch_cpu':torch.get_rng_state()}, 'counters':counters,
             'boundary_only':True}
    checkpoint_io.save(state, path)


def load(path, spec, *, config, rng):
    if not isinstance(rng,np.random.Generator) or type(rng.bit_generator) is not np.random.PCG64:
        raise ValueError('Stage A explicitly supports PCG64 sampler state only')
    state = checkpoint_io.load(path)
    keys={'format','version','spec','source','config_digest','config','versions','components','replay','rng','counters','boundary_only'}
    if not isinstance(state,dict) or set(state)!=keys or state['format']!='hybrid-stage-a' or state['version']!=1 or state['boundary_only'] is not True:
        raise ValueError('Invalid hybrid checkpoint schema')
    if state['spec']!=asdict(spec) or state['source']!=source_fingerprint() or state['versions']!=versions():
        raise ValueError('Checkpoint specification/source/runtime mismatch')
    if state['config_digest']!=config_digest(config) or config_digest(state['config'])!=state['config_digest']:
        raise ValueError('Checkpoint configuration mismatch')
    if not isinstance(state['components'],dict) or not isinstance(state['counters'],dict) or any(type(v) is not int or not 0<=v<=2**63-1 for v in state['counters'].values()):
        raise ValueError('Invalid component/counter schema')
    replay = SequenceReplay.from_state_dict(state['replay'], spec)
    runtime = state['rng']
    if not isinstance(runtime,dict) or set(runtime)!={'python','numpy_global','sampler','torch_cpu'}:
        raise ValueError('Invalid runtime RNG schema')
    # Validate all RNG states in disposable instances before mutating live RNGs.
    probe = random.Random();probe.setstate(runtime['python'])
    numpy_probe = np.random.RandomState();numpy_probe.set_state(runtime['numpy_global'])
    sampler_probe = np.random.default_rng();sampler_probe.bit_generator.state=runtime['sampler']
    torch_probe = torch.Generator(device='cpu');torch_probe.set_state(runtime['torch_cpu'])
    random.setstate(runtime['python']);np.random.set_state(runtime['numpy_global'])
    rng.bit_generator.state=runtime['sampler'];torch.set_rng_state(runtime['torch_cpu'])
    return replay, state['components'], state['counters']
