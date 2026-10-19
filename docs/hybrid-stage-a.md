# Hybrid stage A: data contracts, replay and checkpoints

[Architecture / future acceptance gates](hybrid-draft2.md) · [Reference pins](reference-pins.json) · [Security scope](../SECURITY.md)

Implemented on a separate branch. The existing baseline is unchanged. There is no new RSSM, imagination actor/value learner, continuous Q learner, learned-model planner or distillation implementation here.

## Run from this repository root

Python 3.10+, NumPy, Torch, safetensors and optional Gymnasium are prerequisites. Gymnasium is pinned to the locally available `1.3.0` in `requirements-hybrid.txt` and the `hybrid` package extra. No dependency was installed for this stage. The recommended modern ML dependency combination, clean install and wheel build remain untested; legacy CPU checks are not a security certification.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 60s python -B -m unittest discover -v
python -B -m examples.hybrid_stage_a --output runs/hybrid-stage-a
```

Choose a new output directory for repeat runs. The smoke collects at most three transitions per task in CartPole-v1, Pendulum-v1 and an original deterministic delayed-reward task, saves/resumes a boundary checkpoint for each and writes `report.json`. It performs **zero optimizer updates** and reports no learning/planning claim. It does not render windows, start servers, download models, use a GPU or spend money. These are data-path observations, not comparative benchmark returns.

## Actual contracts

- [contracts.py](../survival_agent/hybrid/contracts.py): `EnvSpec` accepts finite vector observations of declared width 1..1024. `ActionSpec` supports zero-start discrete actions 1..4096 or finite float32 continuous vectors of width 1..256. Images, mixed/infinite spaces and vectorized environments are rejected/deferred.
- Continuous actions in replay/model coordinates lie in `[-1,1]`; affine conversion to/from declared environment bounds is explicit. Invalid actions are rejected before stepping. Discrete masks are exact boolean vectors and constrain real current actions; this is not a planner probability implementation.
- `VectorAdapter.reset(seed=...)` returns the real reset observation. `step(normalized_action)` returns an indexed `Transition` with the actual final observation, separate flags, masks and optional reset observation. Protocol failures require an explicit reset; stale observations cannot be reused.
- Default mode requires explicit reset after a boundary. Optional `autoreset='same_step'` requires `info['final_observation']` or `info['final_obs']` and optional `final_info`; its returned reset observation is stored separately. A wrong/missing autoreset convention is rejected. Synthetic same-step tests do not claim support for every Gymnasium vector/autoreset wrapper.
- `MemorySlots` is lifecycle infrastructure only, not recurrent neural memory. It separates stream slots and clears recurrence/warm-start placeholders on true termination, truncation or explicit reset. A batched/vectorized collector is not implemented.

## Replay indexing and burn-in

[SequenceReplay](../survival_agent/hybrid/replay.py) stores complete episodes: `observations[T+1,D]`, `actions[T,...]`, `rewards[T]`, `terminated[T]`, `truncated[T]`, `is_first[T+1]` and optional `masks[T+1,A]`. Transition `t` connects observations `t` and `t+1`. Continuous actions are normalized; no stale latent states or planner targets are stored.

Start with `start(o_0, mask)` then append each `(a_t,r_t,o_next,terminated,truncated,next_mask)`. At a boundary, the final observation completes this episode; start the next replay episode separately from its reset observation. Appending without start or resetting an unfinished episode fails. Terminal masks may contain no legal action; masks used for actual actions must permit the chosen action.

Capacity eviction removes whole oldest episodes. Transition count, per-episode length, byte budget, exact shapes/dtypes and finite values are checked. Long or incomplete episodes are not silently sliced, joined, padded or checkpointed. The collector must respect the configured horizon/storage bounds.

`sample(batch_size, burn_in=B, unroll=U)` returns independent copied rows of B+U transitions and B+U+1 observations, with loss mask false on burn-in and true on unroll. Default `require_episode_start=True` samples episode prefixes, enabling exact future reconstruction from initial memory. Explicit false allows interior windows and reports `context_is_episode_start=False` and the episode offset; finite burn-in there is an approximation to earlier context. It never invents a reset marker. The caller must choose/label that approximation before learning.

Continuation targets are `1-terminated` with **no gamma folded in**. The return helper rejects interior boundaries and applies gamma once. Truncation bootstraps from the saved final observation; termination suppresses future value while retaining terminal reward.

## Resume boundary and safety

[checkpoints.py](../survival_agent/hybrid/checkpoints.py) reuses the existing bounded structured JSON/safetensors archive. No pickle fallback, dynamic class loading, environment cloning or arbitrary architecture reconstruction was added. Callers provide bounded model/optimizer state dictionaries in `components`; this stage does not construct or train them.

Checkpoints are accepted **only at episode boundaries** with no unfinished replay episode. Arbitrary Gymnasium hidden state and mid-episode environment resume are not supported. Evaluation resumes by reconstructing a fresh explicitly seeded environment and loading caller-created fixed components. This is not evidence that a future learned actor resumes correctly.

Stored state includes bounded replay/config, counters, source fingerprint, runtime versions, model/optimizer component data, replay sampling RNG, Python RNG, NumPy global RNG, PCG64 sampler and Torch CPU RNG. Configuration/source/runtime/spec mismatches fail before live RNG restoration. RNG states are validated in disposable instances first. GPU RNG and cross-version resume are deferred. Loading does not instantiate a model from untrusted checkpoint dimensions.

## Checked acceptance and limitations

- Actual CartPole/Pendulum five-value API, finite vector/action bounds and explicit seeded collection.
- Synthetic same-step final observation differs from reset observation; only the former enters final replay/bootstrapping.
- Terminal and truncation reset both memory placeholders; another stream remains unchanged.
- Exact replay indices, episode-start markers, masks, burn-in/loss masks, independent sample copies and complete-episode eviction.
- Numeric return 3 through truncation vs 2 through termination; interior-boundary and broadcast-shape rejection.
- Structured checkpoint/RNG/sample resume, source/config mismatch and legacy/nonfinite rejection, fixed-component fixed-seed toy evaluation equivalence.
- Optional Gymnasium tests skip explicitly if unavailable; pure data/core tests remain runnable.

Planner isolation/live-state immutability, probability masks, detached distillation targets, actor/model gradient separation and actual learning/comparisons are **future acceptance gates**, not passed tests of this stage. Next is stage B: recurrent stochastic world model plus causal one/multi-step prediction diagnostics, before any imagination learner or planner.
