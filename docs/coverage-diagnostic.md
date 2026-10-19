# Larger matched-budget coverage diagnostic — completed

The historical 249-transition diagnostic is preserved. This follow-up changes
data coverage and sampling only; RSSM architecture, objective and optimizer remain
fixed. No actor, planner, replay-retention mechanism or recycling was implemented.
[Full protocol/report/logs](https://github.com/Nova-Rift/survival-agent-pilot/tree/hybrid-stage-b-survival/docs/experiments/hybrid-coverage-diagnostic).

## Data and frozen comparison

Collected **7,378 real transitions**, 384 training episodes, using four predefined
map/resource configurations and random, greedy, wait and 25%-epsilon-greedy
policies. Config/policy are sampled before the outcome; episodes are never
filtered/resampled to balance classes. Training has **189 deaths and 195 surviving
horizon truncations**. No synthetic-v2 mechanics were used. Old world rules are
unchanged, and each dataset has a recorded content hash.

| Training condition | Episodes | Real transitions | Deaths | Surviving truncations |
|---|---:|---:|---:|---:|
| Historical dataset | 32 | 249 | 6 | 26 |
| Small diverse subset | 16 | 338 | 7 | 9 |
| Full diverse dataset | 384 | 7,378 | 189 | 195 |

The small diverse subset is the first 16 training episodes, chosen without outcome
selection. It helps distinguish variety from scale, but is not an exact same-size
or identical-coverage control. Historical data are retrained from identical seeded
initial parameters under this experiment's budget, not silently compared using
the old smoke's shorter training budget.

Three paired model seeds, three conditions, **300 optimizer updates and 4,800
supervised transition targets per condition/seed** (2,700 updates total). The same
18,639-parameter model, Adam .001, clip 10 and four-target unroll apply everywhere.
Sampling uses the same endpoint/uniform-window mixture in every condition and
reconstructs the actual episode prefix; burn-in is detached. No padding/joining
or approximate interior context. Supervised budgets match, but prefix CPU work
does not: processed context transitions are 8,060..8,200 historical,
20,060..20,748 small diverse, 17,636..18,676 full diverse. Training run times were
about 2.7..4.1 s each; total collection/training/evaluation about **63.6 s** CPU.
Training is endpoint-focused, and actual target class exposures are logged;
**evaluation retains natural class prevalence**.

Training map seeds 30000..30383, development 40000..40063, final 50000..50127 are
disjoint; historical training seeds remain 0..31. Architecture/objective and all
conditions were predeclared. Development was scored, no changes selected, then
model/dataset hashes and protocol were frozen **before** final collection.
Previous final seeds/results remain historical. The frozen file SHA-256 is
`d11b712be0bd6d21993ec9e008107e55d2a9afb55f908f6e0a9db84b630d48a4`.

Development: 64 episodes, 1,137 transitions, 40 deaths/24 surviving truncations.
Final: **128 episodes, 2,449 transitions, 68 deaths/60 surviving truncations**.
Every final episode is scored once. After one real context transition per episode,
there are 2,321 future prediction targets, including all 68 deaths. Future targets
are never fed to the prior fork; recorded actions are. Categorical-mode rollouts
are approximations, not stochastic belief averaging.

## Final evidence

Means across three initialization seeds (all individual results retained):

| Condition | Prior reward MSE | Own training-mean baseline MSE | Prior continuation BCE | Baseline BCE | Prior death recall at .5 |
|---|---:|---:|---:|---:|---:|
| Historical | .03486 | .02847 | .13765 | .13283 | 0% |
| Small diverse | .04103 | .02851 | .17443 | .13391 | 12.75% |
| Full diverse | .02872 | .02845 | .14551 | .13255 | **0%** |

Full-data prior reward MSE ranges .02804..02906, close to a trivial mean predictor.
It does not consistently outperform that baseline, and continuation BCE is worse
for all three seeds. On actual deaths, full-data priors predict continuation
.9885..9927; their mean death-conditional Brier is **.98085**. The low overall
Brier around .0288 and high aggregate accuracy mostly reflect the numerous
continuing transitions. Full-data posterior reward MSE averages **.00213**,
posterior death recall is 100% for all seeds, and posterior continuation on deaths
is .0483..0647. That posterior has already seen the final observation, so this is
outcome recognition, **not advance prediction of failure**.

Horizon/example counts for full-data seed 0:

| Future horizon | Eligible targets | Deaths | Prior mean continuation | Actual continuation |
|---|---:|---:|---:|---:|
| 1 | 128 | 0 | .9901 | 1.0000 |
| 3 | 128 | 0 | .9896 | 1.0000 |
| 7 | 128 | 23 | .9889 | .8203 |
| 11 | 102 | 18 | .9888 | .8235 |
| 15 | 80 | 10 | .9890 | .8750 |
| 23 | 62 | 0 | .9891 | 1.0000 |
| 31 | 19 | 0 | .9891 | 1.0000 |

All horizons and calibration bins, conditional Brier/recall and posterior/prior
gaps are reported. Later horizons exclude episodes already ended; do not interpret
their improved prevalence as evidence of better death prediction. Full-data
posterior-to-open-loop-prior KL averages .91..1.06 nats across seeds. One seed has
zero memory divergence but nonzero categorical/head gaps; recurrent drift alone
is not a complete explanation. Development similarly has zero full-data prior
death recall, so the final outcome was not unique to one reporting set.

## Decision and limits

**Do not proceed to learned-model planning on this evidence.** More representative
data and matched target/update budgets improved some aggregate reward errors,
but prior death calibration remains inadequate. Data coverage alone under this
fixed budget is not enough. This does not prove insufficient capacity: the model
is tiny, only 4,800 supervised targets are used per condition, mixture/history
lengths vary, and a single mode approximates stochastic dynamics. Posterior-only
head fitting, free-nat pressure and objectives that insufficiently constrain
open-loop failure prediction remain hypotheses, not isolated causes. No new
architecture/objective was justified or implemented. Inspect existing transition
prediction and use development-only follow-up tests before choosing a model change.

Added regression coverage verifies fixed loss-target budgets, exact copied prefix
context/endpoints, no dataset eviction, deterministic data hashes, natural outcome
accounting and disjoint new final seeds. Core calibration fixtures check death
and survival conditional scores. **32 agent and 61 integration tests pass**.
No new packages, paid compute, GPU or trained survival policy. Legacy runtime
security caveats remain. No survival-improvement or continual-learning claim.
