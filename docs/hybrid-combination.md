# Combined model implementation — October 19, 2026

The requested combination is now an executable experimental path, separate from
historical GRU and posterior-only Transformer baselines. All neural architecture
is in [`model.py`](../survival_agent/hybrid/model.py), class `HybridModel`.
The supplied private design is not redistributed. This document records the
implementation and remaining research gates, not completion of every evaluation
in that design.

## Responsibilities

- `model.py`: Transformer world model, categorical prior/posterior, prediction
  heads, discrete or continuous actor, state value, continuous Q ensemble.
- `policy.py`: categorical and tanh-Gaussian distributions, sampling/log probabilities.
- `learning.py`: real-prefix reconstruction, world fitting, imagination, TD,
  slow targets and optional distillation. Defaults leave distillation disabled.
- `planning.py`: MCTS and weighted MPC, frozen model copies per call, no environment
  references. Explicit computation budgets; categorical-mode planning approximation.
- `runtime.py`: separate state/action/warm-start per stream and reset on either
  termination or truncation. `auto` dispatches MCTS for discrete, MPC for continuous;
  planning disabled selects the actor. Actions remain normalized until the collector.
- Integration repository `survival_adapters/hybrid_combination.py`: actual Gymnasium
  stepping, replay, bounded training/evaluation, reports and safe boundary checkpoints.

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
```

From the sibling integration checkout:

```bash
PYTHONPATH=../survival-env:../survival-agent OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -B -m examples.hybrid_combination --task CartPole-v1 --output /private/new/cartpole
PYTHONPATH=../survival-env:../survival-agent OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -B -m examples.hybrid_combination --task Pendulum-v1 --distillation --output /private/new/pendulum
```

Each command has four collection episodes capped at 16 steps, four update cycles,
and one short evaluation episode each for actor and planner on the same frozen
checkpoint. Configuration and source hashes are recorded before collection. It
saves replay, weights, slow targets, optimizers, normalization and RNG at an episode
boundary using the safe serializer. It does not support mid-episode environment
resume. A new output directory is mandatory; historical artifacts are not replaced.

## Verification and limits

Checks: 50 model tests and 71 integration tests pass, including eight new hybrid
tests in the model suite. Known tiny
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
