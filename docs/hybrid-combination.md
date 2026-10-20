# Combined model implementation — October 19, 2026

The requested combination is now an executable experimental path, separate from
historical GRU and posterior-only Transformer baselines. All neural architecture
is in [`model.py`](../survival_agent/hybrid/model.py), class `HybridModel`; the same
file now also contains action distributions, MCTS and MPC at Clark's request.
The supplied private design is not redistributed. This document records the
implementation and remaining research gates, not completion of every evaluation
in that design.

## One-file layout

Clark requested that everything needed for a full training loop be readable in
`model.py`. The file is ordered as follows:

1. Neural architecture: Transformer/categorical world model, policy, value and Q heads.
2. Action distributions and MuZero-style MCTS / TD-MPC2-style MPC.
3. Complete-episode `SequenceReplay` and world-model sequence fitting.
4. `HybridLearner`: imagination, real TD, optional distillation and optimizer state.
5. `HybridController`: per-stream memory, episode resets and planner dispatch.
6. `TrainingSession`: exploration, collection, updates, evaluation and checkpoint resume.
7. `main()`: runnable CPU training command.

`planning.py`, `learning.py`, `policy.py`, `replay.py` and `runtime.py` are now
compatibility imports, not additional implementations. `world_model.py` preserves
the old GRU architecture and re-exports the shared training helpers from `model.py`.
Environment definitions remain external. Generic shape/adapter contracts and the
existing pickle-free serializer are supporting dependencies; no training algorithm
must be read from another file. No environment is loaded or training started on import.

The old `transformer_agent.py` remains the posterior-only baseline, not an alias
silently redirecting old experiments to a new algorithm. Old checkpoint source
checks remain strict. Added Q/nonlinear policy heads require new hybrid weights;
no old checkpoint is claimed to contain them. The old integration fingerprint now
also includes `model.py`, closing the provenance gap introduced by the file split.

## Exact algorithm choices and deviations

Dreamer-inspired learning starts from the posterior before the final transition of
each sampled real episode-prefix row. Full supplied context is reconstructed; no
random latent starts or reset-observation bootstraps. Imagined trajectories use the
prior only. Both policies use a score-function objective with detached actions,
features, advantages and cumulative continuation weights. Scalar state values use
raw-unit MSE with slow-value regularization. Return scaling tracks exponential
5th/95th percentiles with a floor of one. Observation/reward training uses symlog;
search and imagined rewards invert it. KL balancing, free nats and categorical
uniform mixing retain the existing implementation. These reduced scalar heads,
normalization and separate optimizers differ from full DreamerV3 distributions
and optimization. No policy/Q gradients enter the world model.

[Inspected pinned Dreamer policy loss](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/dreamerv3/agent.py).

MCTS uses policy priors, the MuZero logarithmic exploration coefficient, running
min/max Q normalization, cached deterministic transitions and single-agent backups.
Actual legality is root-only; imagined interiors are explicitly unmasked. Training
can add Dirichlet root noise and sample visit counts with a temperature. Evaluation
chooses the highest visit count. Continuation gates only future returns; gamma is
separate. This is MuZero-inspired planning, not original MuZero's unrolled
policy/value training objective or chance-aware search.

[MuZero primary reference](https://arxiv.org/html/1911.08265v2).

Continuous TD uses the minimum of two random target Q heads and the actual next
posterior from real replay. Truncation keeps its bootstrap; termination removes it.
MPC uses actor proposals, shifted previous means, bounded Gaussian candidates,
exponentially weighted elites and weighted variance updates. It uses the mean of
two online Q heads (fixed pair during one search) at an actor-mean horizon action;
state value is an explicit ablation. Evaluation picks the best elite; training
samples elites and adds bounded exploration. These are documented simplifications
of TD-MPC2. Unlike its decoder-free joint objective, representation gradients stop;
there are no SimNorm, distributional Q heads, task embeddings or Q-trained actor
objective. Q and state-value critics remain distinct.

[Inspected pinned TD-MPC2 implementation](https://github.com/nicklashansen/tdmpc2/blob/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/tdmpc2.py).

Distillation targets store copied detached tensors, exact real-history/mask digest,
planner settings, model revision and learner step. They are accepted only for the
same current world-model revision and bounded age. Discrete visits train masked
cross entropy; continuous weighted first actions train actor-mean MSE. Targets are
not declared good teachers. The no-distillation path is the default.

## Commands

From the model checkout, using already available dependencies:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -B -m unittest discover -v
python3 -B -m survival_agent.hybrid.model --task CartPole-v1 --output /private/new/cartpole
python3 -B -m survival_agent.hybrid.model --task Pendulum-v1 --distillation --output /private/new/pendulum
```

The full loop first collects exploratory episodes, then alternates real-replay
updates and actor/planner collection. Updates fit the world model, train imagined
policy/value returns, optionally distill planner targets, and train continuous Qs.
Evaluation uses the same frozen checkpoint with actor/planner modes and restores
training RNG state afterward. Every cycle writes metrics and a safe checkpoint.
The collector's step cap is truncation and preserves final-observation bootstrapping.

Defaults: two exploratory episodes, four cycles, one collected episode and two
update batches per cycle, episodes capped at 16 steps. `TrainingConfig` exposes
collection/update ratios, batch size, prefix length, replay capacity and evaluation
count. `--mode actor` disables planning. Distillation remains off by default.

To interrupt and resume at a cycle boundary, keep the original total configuration:

```bash
python3 -B -m survival_agent.hybrid.model --cycles 4 --run-cycles 2 --output /private/new/part1
python3 -B -m survival_agent.hybrid.model --cycles 4 --resume /private/new/part1/checkpoint.safe --output /private/new/part2
```

Saved state includes replay, weights, targets, optimizers, normalization, RNG and
collection/cycle counters. Source, runtime and configuration checks remain strict.
There is no mid-episode environment resume. Output directories must be new.
The earlier integration command remains available as a compatibility workflow;
the primary full-loop command now runs directly from the model repository.

## Verification and limits

Checks after consolidation: 52 model tests and 71 integration tests pass. The
new full-loop test verifies both action types, batched distillation, collector
truncation and exact equality between uninterrupted and resumed updates/weights.
Compatibility imports resolve to the same implementation. Actual CLI checks
complete a CartPole partial run/resume and a Pendulum run with distillation.
Earlier hybrid checks are recorded below. Known tiny
models verify delayed-reward MCTS and bounded MPC choices. Actual neural-model
search does not decode observations or mutate live tensors/weights. Optimizer/RNG
checkpoint resume reproduces the next update exactly; restored model planning
reproduces a fixed-RNG action. The return regression gives 3 through truncation
and 2 through termination for the specified rewards/discount/bootstrap.

Bounded integration checks used 63 CartPole and 64 Pendulum real training
transitions, four world and four imagination updates each, and four continuous
TD updates on Pendulum. Optional distillation was exercised for both. Single
frozen-checkpoint episode returns: CartPole actor 9, MCTS 16; Pendulum actor
-118.8009, MPC -114.6897. These tiny runs are **not learning-improvement evidence**
or controlled comparisons. The initial Pendulum check found weighted-mean
floating-point overshoot in warm-start validation; clamping fixed it and repeated
decision tests were added. The failed artifact is preserved alongside the corrected
run. No scientific checkpoint was selected from these checks.

CPU only, no dependencies installed, paid compute or social-game inference.
Public website/diagram and research survival controllers remain unchanged.
Multi-seed actor-learning evidence, learned-model calibration, stochastic/chance-aware
planning, matched-budget comparisons, performance/retention guarantees and the
broader continual-learning objective remain open. The full specification's research
definition of done is not claimed by this implementation milestone.
