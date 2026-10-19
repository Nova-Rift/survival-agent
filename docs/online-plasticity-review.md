# Architecture inspection before plasticity changes

Inspection baseline: model `67913fe3d1f1cae4a28ca4c8647b1b75e71aac4f`,
integration `63a8869416f3ab56416453e89ee7f5986941aced`. This is a recommendation,
not an implemented replay/TD/recycling redesign. No recycling code was added.

Standing research goals are learning during gameplay, retaining older skills, and
preserving the ability to keep learning. This note preserves the proposal and
reference for later experiments; none of the proposed plasticity changes is
implemented or adopted on the strength of the paper alone.

## What exists

- `survival_agent/agent.py:12` (`HistoryAgent`): local history embedding and one
  Transformer encoder, a small action-conditioned transition MLP and reward head,
  policy head and **one linear value head** (`value`, line 24). `smoke_update`
  (line 68) already has observed-transition reward and latent-consistency losses,
  one-step value bootstrapping and greedy imitation; it is not validated control.
- `survival_agent/integrated.py:16` (`LanguageLatentCore`): shared feature
  projection, transition MLP, reward/policy heads and **one linear value head**
  (line 27). `LocalLanguageBackend` is CPU/eval/frozen at
  `survival_agent/language_backend.py:101`; its `encode` uses no gradients.
- `survival_adapters/training.py:61` (`Trainer`, in the integration repo) is the
  existing during-play actor-critic path. It collects fresh rollouts (default four
  joint steps), updates after collection, and continues gameplay with new weights.
  The frozen language backend stays outside the optimizer. **No replay buffer**
  exists in this trainer; only its latest rollout contributes.
- Its `survival_targets` (line 48) retains the received reward, suppresses bootstrap
  on death, bootstraps truncation/cutoff from actual next features and cuts reset
  recursion. `collect` (line 76) records the decision before encoding next local
  experience and captures final next features before a later explicit reset.
  Inactive slots are masked. Current settings are gamma .99, target EMA .99,
  Adam .001, gradient clip 1. The target is a frozen copy of the **whole core**.
- That trainer has one optimizer over **all trainable core parameters**: actor,
  value and observed reward/dynamics losses share projection/transition gradients.
  It has no isolated value optimizer or recycling moment/target reset logic.
  Global episode reset clears controller histories/call counters. Dead slots are
  excluded from use, but their stored histories are only cleared at global reset;
  do not describe this as per-death clearing. No persistent neural RSSM is used.
- `survival_agent/muzero.py:137` (`EpisodeReplay`) retains the newest 128 episodes,
  then samples retained episodes and positions uniformly. Its n-step value targets
  use stored search root values; it is a separate minimal MCTS/unroll path, not the
  actor-critic trainer. It does not reserve representative older tasks after FIFO
  eviction, and has no learned target critic ensemble.
- `survival_agent/hybrid/replay.py:14` (`SequenceReplay`) evicts whole oldest
  episodes under step/byte bounds. Default sampling chooses eligible episode
  prefixes; optional interior context is labeled approximate. There is no
  recent/old partition, reservoir, age quota or high-error priority. The current
  coverage diagnostic adds explicit exact-context sampling in the integration
  repo; that is not a continual-memory mechanism.
- `survival_agent/hybrid/world_model.py:58` (`WorldModel`) is the implemented
  categorical RSSM with GRU/posterior/prior/reconstruction/reward/continuation.
  `WorldModelTrainer` (line 248) updates only model parameters on collected real
  replay. It has no actor, value/Q critic or gameplay learner. Stage-A
  `MemorySlots` and `VectorAdapter` reset stream memory on both boundary flags.
  There is no unified RSSM online-policy/TD loop to extend by just toggling a flag.

## Architecture-specific recommendation

Do not add another full world model. Assess the transition/reward losses already
present in the actor-critic path, and the existing RSSM's posterior/prior gap.
The larger coverage diagnostic still fails to predict death reliably in prior
rollouts; neither better reconstruction nor a fit posterior justifies search.
Optional prediction auxiliaries should be explicit later ablations, not assumed
necessary for plasticity. Defer modules, fast/slow systems and intrinsic rewards.

First add a bounded mixture of **recent transitions and representative older
training replay** to the value learner, with fixed documented quotas (initial
proposal 75% recent / 25% older). Use task/context-stratified reservoir retention or equivalent
coverage accounting, not only recent/high-error transitions. Keep raw local
history or provenance-bound frozen features plus actual next observation,
separate termination/truncation and episode/task identifiers. Never retain stale
RSSM latent states as permanent training inputs. Hold-out validation is not replay.

Replay must not simply be fed into the existing on-policy actor loss: older
actions came from older policies, and this would silently introduce off-policy
policy-gradient bias. In the first isolation experiment, use old/recent replay
for **TD value updates only**, keep actor updates on fresh rollouts, and hold any
auxiliary prediction objective fixed across arms. A separate later algorithm
decision is needed for off-policy policy learning.

TD target proposal: `r + gamma*(1-terminated)*V_target(actual_next_features)`,
under no-grad, with a separate frozen target value network and fixed EMA. A
time-limit cutoff bootstraps its final observation; a reset observation must never
replace it. TD improves credit assignment conditional on observed experience;
it does **not** solve sparse rewards or exploration. Reuse the existing boundary
fixtures before changing collection or optimizer code.

## No existing eligible hidden value layer

Both existing value heads are `Linear(width, 1)`. Recycling their input coordinates
would alter shared features; recycling projection/transition/Transformer units
would violate the requested exclusions. There is therefore **no existing hidden
feedforward value-only layer to recycle safely**.

Smallest candidate is a flag-controlled, private two-layer value MLP, e.g. frozen
local backend features -> Linear(32) -> ReLU -> Linear(1), owned in the model repo
and integrated by the adapter trainer. Feed it detached/frozen features and give
it a value-only optimizer and target copy. Exclude the language backend, shared
projection, transition model, RSSM encoder/decoder/heads, GRU and history memory.
This changes critic capacity, so the unmodified linear critic must remain an
anchor; all recycling comparison arms need the **same MLP**. Do not compare a new
MLP+recycling only against an old linear head and attribute the difference to
recycling.

## Smallest useful experiment before broad integration

First run a critic-only repeated A -> B -> C -> A streaming pilot on fixed real survival replay,
with identical behavior data, targets, recent/old quotas and update cadence.
This tests value plasticity/retention cheaply; it does not demonstrate a better
gameplay policy. Use five paired initialization seeds, distinguishable scenario context, and matched transitions,
optimizer steps, widths, batch size and diagnostic frequency. Proposed small
pilot budget: measure head-update throughput first, then set a ceiling of 100,000
head updates total across all arms/seeds, including anchors; start smaller if the
CPU/runtime budget requires it. Record real stream transitions, update cadence,
CPU time and total recent/historical replay exposures.
Preserve the current environment version; synthetic-v2 is a separate later suite.

Report an unmodified-linear anchor plus three architecture-matched arms:

1. Private value MLP, no regularization, recycling disabled.
2. Identical MLP with mild value-weight regularization only.
3. Identical regularization plus flag-controlled recycling.

Choose regularization strength, maturity/utility thresholds and replacement rate
only on development streams, then freeze before fresh final streams. Never recycle
based only on current activation or error. Track usefulness on **both recent and
representative older training replay**; veto replacement if historical usefulness
or an old-replay unit-ablation check indicates support for an older skill. If no
unit meets conservative criteria, replace none. This is a testable safeguard,
not a proof against forgetting. The reviewed starting utility is mean absolute activation times outgoing-weight
magnitude, evaluated on recent and historical probes. Paper examples such as rate
1e-4, maturity 10,000 and utility EMA .99 are candidate development settings, not
universal defaults. No fidelity claim is made here.

An eventual implementation must initialize only the selected private units,
zero bias/outgoing connections with explicit before/after historical-probe output
change checks, and reset their
Adam moments and utility/maturity counters, and explicitly reconcile matching
target-network units. Do not reset histories, model memory, other optimizer slots
or pretrained weights. Verify disabled-flag equivalence, optimizer/gradient
isolation, reset/resume reproducibility and rejection of historically useful units.

## Reference-review constraints

The separately reviewed [Nature study](https://www.nature.com/articles/s41586-024-07711-7)
and [official code](https://github.com/shibhansh/loss-of-plasticity) motivate testing
plasticity; they do not guarantee retention of older skills. Published RL settings
use different L2 weights for L2-only (1e-3) and CBP+L2 (1e-4), so the decisive local
comparison must use **identical regularization** in arms two and three.

Optimizer semantics must be specified rather than assumed: generic upstream GnT
resets moments and per-weight bias-correction ages when using custom `AdamGnT`,
whereas the PPO path passes `torch.optim.Adam`. Do not claim universal optimizer
reset behavior from the paper code. For the proposed pilot, use the **same Adam
implementation in every arm** and explicitly choose/test selective recycled-slice
moment resets. Standard Adam has a per-parameter step count, not per-weight ages;
keeping that count while zeroing selected moments is not equivalent to AdamGnT's
per-weight reset. If per-weight ages are required, that is a separately reviewed
optimizer implementation shared by every arm, not a recycling-only advantage.
Zeroing outgoing weights removes the old contribution: recycling is **not function
preserving**. Never reset the whole shared optimizer. Pin the official revision and check its
license before implementing or borrowing algorithm code.

## Predeclare success criteria

- Before the pilot: all boundary/isolation/reset tests pass; recycling disabled
  reproduces the identical MLP control; only the private value component changes.
- Critic-only screening proposal: at matched update/transition budgets, recycling
  must reduce new-task held-out return-prediction error learning-curve area by at least 10%
versus the
  regularized-only arm, while old-task held-out return-prediction error rises no more than 5% relative to that
  arm. Require the direction in at least four of five paired pilot seeds. If the plain
baseline shows no loss of plasticity, the recycling question is inconclusive,
not a success. Do not use TD training loss alone as the acceptance metric. These are
  proposed practical gates, not achieved results or statistically strong evidence.
- Before adoption: confirm with at least five seeds in the actual during-play
  actor-critic path, identical policy exploration and budgets. Primary measures
  are new-skill learning speed, old-world survival retention, final survival,
  continued learnability across repeated shifts, and CPU/memory cost. Predeclare
  normalization and retention tolerances on development data. Report paired
  seed-level uncertainty and negative results; TD error alone is insufficient.
- For synthetic-v2, normalize alive-step return by `min(horizon, wear_limit-1)`
  under the proposed post-transition reward convention and a wear limit >=2.
  Report true wear failure separately from horizon truncation. No comparison may
  silently mix legacy and new environment versions.

The cited Nature page could not be retrieved directly in this executor; the
completed separate paper/code review supplied the reference constraints above. The official
repository was retrieved, but no upstream algorithm was installed or executed.
No replay redesign, TD probe, regularization or recycling experiment was run in
this inspection milestone. The online baseline itself has no validated continual
learning result. Adoption requires evidence, not a new architecture diagram.
