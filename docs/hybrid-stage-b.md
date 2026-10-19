# Stage B: small recurrent stochastic world model

[Follow-up diagnostic: broader coverage, trivial baselines and posterior/prior gaps](hybrid-stage-b-diagnostics.md)

Original PyTorch vector RSSM in
[`world_model.py`](../survival_agent/hybrid/world_model.py), built on stage A.
The stage-A commit and the legacy models remain intact. This milestone trains
prediction heads from real replay; it implements no actor, value learner,
imagination-policy learning, MPC, MCTS or distillation. It establishes no survival
improvement, retention or planning benefit.

## Model and causal indexing

The model has a symlog vector encoder, `GRUCell` deterministic memory, grouped
categorical stochastic state, prior, observation-conditioned posterior, symlog
observation decoder, symlog reward regression and Bernoulli continuation head.
The CPU smoke uses 32 memory units, 32 hidden units and four four-class latent
variables: **18,639 parameters** for 45 observation features / five actions.

`observe_step(previous_state, previous_action, current_observation, is_first)`
resets both memory components and previous-action features independently for
each flagged stream, then updates recurrence and infers the posterior. The
prior sees only the resulting recurrent memory, never the current observation.
`prior_step(state, action)` predicts a new latent without reading observations,
decoding images/vectors or accessing an environment. `heads(decode=False)` makes
reward/continuation available without invoking the decoder. Input state is not
mutated. Actual interaction must call `observe_step` for every real observation;
this stage's collector is scripted and does not yet carry an online learned policy.

Replay transition t links `observations[t]` to `observations[t+1]` via
`actions[t]`. Training first infers the initial observation, then posterior
state t+1 reconstructs observation t+1 and predicts reward t and continuation t.
Initial-state reconstruction/KL has no separate loss in this small milestone.
Continuation is exactly `1 - terminated[t]`, with **no gamma included**.
Truncation retains continuation; either boundary resets interaction memory for
the next episode. Stored final observations are retained before explicit reset.

## Loss, stabilization and simplifications

Reviewed upstream reference at pinned DreamerV3 revision
`e01491fad6434b2245a3b8ca201dd7faedcc458c`:
[`rssm.py`](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/dreamerv3/rssm.py)
and [`configs.yaml`](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/dreamerv3/configs.yaml).
These inform the following choices; no upstream source was copied:

- Grouped categorical posterior/prior, straight-through one-hot samples and
  0.01 uniform probability mixing. Training samples are stochastic. Diagnostics
  use categorical modes, an explicit approximation to stochastic predictions.
- Dynamics loss `KL(stopgrad(posterior) || prior)` and representation loss
  `KL(posterior || stopgrad(prior))`. Sum category KL across stochastic variables
  first, then clamp each loss to one free nat per state. Weights are 1.0 and 0.1.
  Below the threshold the clamped KL has zero gradient, not a zero logged loss.
- Reconstruction and reward use squared error in symlog space; continuation uses
  binary cross-entropy with logits. Losses average over vector features, batch
  and unroll. `symexp` converts diagnostic predictions back to observation/reward
  units. Do not add symlog rewards directly to future return calculations.
- A small standard PyTorch GRU replaces upstream block recurrence. LayerNorm
  replaces RMS normalization; architecture/initialization/optimizer differ.
  Scalar symlog reward regression replaces the upstream two-hot reward head.
  Feature-mean reconstruction differs from upstream summed likelihood losses.
  Adam, fixed learning rate and global norm clipping replace upstream optimizer
  details. These are documented simplifications, not DreamerV3 reproduction.
- Return/value normalization is deferred to actor/value learning in stage C;
  there are no return/value targets in this world-model-only milestone.

## Replay training and bounds

`sequence_loss(model, rows)` consumes copied stage-A `SequenceReplay.sample`
rows. It requires true episode-prefix context, exactly shaped finite arrays,
normalized continuous actions or valid discrete actions, a single reset at the
start, no interior boundary, and a contiguous burn-in/unroll loss mask.
Approximate interior-window context is rejected in this stage. Initial context
and burn-in run without gradients; memory is detached at the unroll boundary.
Future unroll steps backpropagate through recurrence. No padded/joined episodes,
stale saved latents or planner targets are used. Reward/continuation targets come
only from real replay.

`WorldModelTrainer.step` performs one stochastic sequence update, checks finite
loss/gradient norm, clips gradients and updates only the model's parameters.
Collection-to-update ratio is explicit in the integration smoke: 16 complete
training episodes collected first, then 40 updates of batch four, burn-in one,
unroll three. It does not assume one update per environment step.

Bounds: vector observations 1..1024; encoded actions at most 256; deterministic
and hidden widths 8..256; 1..16 categorical variables, 2..32 classes; batch at
most 32; burn-in at most 128; unroll at most 256; batch×sequence at most 4096.
Optimizer rate is at most .01 and clip norm at most 100. The sequence trainer is
explicitly CPU-only. Very large discrete action spaces accepted by stage A are
deferred beyond encoded width 256. No image/spatial encoder is implemented;
the survival adapter explicitly flattens one local 5×3×3 observation to a vector.

## Tests and commands

From this model checkout, with already installed dependencies:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 60s python -B -m unittest discover -v
```

**29 tests passed**, including all 18 prior tests and 11 new stage-B tests.
New checks cover discrete/continuous shapes, finite gradients for all components,
causal prior inputs and observation-sensitive posterior, persistent recurrence,
independent stream resets and state immutability, balanced-KL gradient stops,
uniform mixing/free-nat floor, transform round trip, exact next-observation and
reward/flag indices, no gradient through burn-in, prior-only future diagnostics,
bounded invalid data, tiny fixture fitting, and safe checkpoint restoration of
model/optimizer/replay/RNG with an exactly matching next stochastic update.

Adapters and real-world smoke belong in sibling `survival-agent-pilot-source`,
branch `hybrid-stage-b-survival`, **not** this core package. With this model branch
and the existing environment checkout as siblings:

```sh
PYTHONPATH=../survival-env:../survival-agent OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 60s python -B -m examples.hybrid_stage_b --output runs/hybrid-stage-b --updates 40
```

Choose a fresh output directory. The integration repo's published dependency
pins do not include this branch yet; use the explicit sibling workflow.
The smoke creates structured `.safe` model/optimizer/replay/RNG checkpoints,
`updates.jsonl` and `report.json`; it never calls `torch.load` or pickle. The caller
constructs the fixed bounded model; checkpoint loading does not construct an
architecture. The config binds the world-model source SHA-256 in addition to the
stage-A source fingerprint. Resume remains episode-boundary and same-runtime.

## Actual CPU smoke evidence and limits

One seed, 16 scripted random/local-greedy training episodes (79 transitions) and
four separate held-out seeds (20 transitions), then 40 model updates. Fixed
evaluation rows are sampled with replacement from their respective episode sets.
Training never uses held-out episodes. Node/browser inference is not part of this
milestone. Python 3.10.12, NumPy 1.26.4, Torch 2.2.0+cu121 and safetensors 0.4.5
were already installed; no package install, GPU, model download or paid compute.
Known legacy-runtime advisory caveats remain; functional checks are not security
certification. The 40-update collection/training/checkpoint run took about 0.84 s
on the local CPU with one Torch thread; this is not a cross-machine benchmark.
All collected episodes ended in death; two training and one held-out episode
also reached the horizon. Exclusive-truncation semantics have unit-test coverage
but no data coverage in this particular smoke collection.

| Fixed mode-latent diagnostic | Before | After 40 updates |
|---|---:|---:|
| Training total prediction loss | 2.03367 | 1.40139 |
| Held-out total prediction loss | 2.04366 | 1.55207 |
| Held-out one-step prior observation MSE | 0.60752 | 0.09457 |
| Held-out three-step prior observation MSE | 0.33719 | 0.10183 |
| Held-out one-step prior reward MSE | 0.36895 | 0.03762 |
| Held-out three-step prior reward MSE | 0.14791 | 0.58644 |
| Held-out three-step continuation BCE | 0.67527 | 1.46046 |

The **three-step reward and continuation predictions worsened** despite better
reconstruction and total teacher-forced loss. This small model is not validated
for planning. Loss includes the free-nat floor, so it cannot be interpreted as a
raw prediction-error score. Diagnostics condition on one real transition after
the initial observation, then use only priors and actual recorded actions for
one/three future steps; future observations are error targets only.

Safe-checkpoint fixed diagnostics matched exactly. The integration suite passed
56 tests. Full update logs and a source/config/runtime report are retained in the
integration repo's `docs/experiments/hybrid-stage-b-seed0/` directory, without
weights or private paths. Single-seed tiny-subset errors are engineering evidence
only. Next work must investigate open-loop reward/termination prediction,
broader held-out data and stronger baselines before relying on the model for
imagination learning or search. Controlled survival/retention comparisons,
multiple-seed learning evidence and all later Draft-2 gates remain unfinished.
