# Stage-B open-loop diagnostic

The original stage-B commit/report are preserved. No RSSM equations, model loss,
legacy baseline or default smoke command was changed by this diagnostic.
[`diagnostics.py`](../survival_agent/hybrid/diagnostics.py) infers a real prefix,
then forks independent prior-only and teacher-forced posterior paths at the same
transition indices. Only the posterior fork receives future observations. Neither
fork accesses an environment. Categorical modes approximate stochastic beliefs;
these are not calibrated integrals over all latent outcomes.

The integration branch provides complete-episode batching and a fixed CPU protocol.
[Protocol, report and logs](https://github.com/Nova-Rift/survival-agent-pilot/tree/hybrid-stage-b-survival/docs/experiments/hybrid-stage-b-diagnostic).
Final seeds were not used for gradients, hyperparameter or condition selection.
All six trained conditions are reported, including negative outcomes.

## Fixed experiment and class counts

Same 18,639-parameter model, rate .001, model seeds 0, 1, 2, 80 updates per
condition, batch four, burn-in one. Compare the existing three-transition prefix
loss with complete-episode losses. Architecture, initialization seed, optimizer,
collector mix and real dataset match. **Supervised transition counts and compute
do not match**: this is an exposure diagnostic, not a matched-compute causal
ablation. There are 480 total updates, about 7.4 s on one local CPU thread.

Unchanged mechanics: six-by-six grid, one agent, radius one, hunger six, six
initial food cells, one spawned food per step, horizon eight. Collection alternates
random/local-greedy scripted policies. Training seeds 0..31 (249 transitions),
development 1000..1015 (120), final 2000..2015 (120) are disjoint. The plan is saved
before final collection; all optimization finishes before final data are collected.
Each held-out episode is evaluated once, without replacement sampling.

| Split | Episodes | Death endpoints | Exclusive truncations | Both death + horizon |
|---|---:|---:|---:|---:|
| Training | 32 | 6 | 26 | 2 |
| Development | 16 | 4 | 12 | 0 |
| Final | 16 | 4 | 12 | 0 |

The both column is a subset of death endpoints. Continuation is zero on death
even when horizon also occurs, and one on exclusive truncation. In this unchanged
one-agent task, reward equals post-transition alive status, so reward and
continuation targets are identical; that identity is checked against real replay.

## Coverage finding and bounded remedy

Reproducing the original smoke gives 79 transitions. Its unique episode-prefix
loss targets contain nine death labels among 48 supervised transitions, versus
16 among 63 complete-episode targets (both exclude first-transition burn-in).
There are no exclusive truncations. Thus the original result does **not** show
absence of death labels; it shows narrow coverage and uncalibrated predictions.

In the broader hunger-six task, those same four-transition prefixes contain zero
death and zero horizon labels: all 96 unique targets continue. This is a verified
**training-window coverage limitation**, not an off-by-one loss or label defect.
Complete-episode sampling remedies exposure while preserving real final
observations, episode-start context and endpoints, without padding/joining or
changing mechanics. Length groups are weighted by episode count, then same-length
episodes are sampled with replacement. Actual training exposures:

| Model seed | Prefix supervised / death / exclusive truncation | Complete supervised / death / exclusive truncation |
|---|---|---|
| 0 | 960 / 0 / 0 | 2196 / 51 / 269 |
| 1 | 960 / 0 / 0 | 2176 / 53 / 267 |
| 2 | 960 / 0 / 0 | 2156 / 71 / 249 |

## Scores and calibration

Final scoring begins after one real transition: 104 future targets, four deaths
and 100 continuations, including 12 exclusive truncations. The constant baseline
is fitted only on all real training labels. Final training-mean baseline reward
MSE is **.03719**, continuation BCE **.16675**, Brier **.03719**. The report also
includes always-alive and untrained baselines, endpoint-only and horizon-wise
scores, counts, calibration bins and ECE. BCE clips probabilities to 1e-6 only
for logarithms; Brier/calibration use unclipped probabilities.

Final means ± population SD across three initialization seeds:

| Condition / inference | Reward MSE | Continuation BCE | Continuation Brier |
|---|---:|---:|---:|
| Prefix / prior | .04249 ± .00328 | .16540 ± .00124 | .03720 ± .00013 |
| Complete / prior | .04718 ± .00794 | .16745 ± .00128 | .03742 ± .00014 |
| Complete / posterior | .02476 ± .00220 | .15617 ± .00120 | .03605 ± .00021 |

Complete-episode prior reward MSE ranges **.03822..05753**. Neither condition
consistently beats the trivial baseline for open-loop reward. Complete exposure
alone does not establish better open-loop continuation. Development has the same
pattern (complete/prior reward MSE .04758, complete/posterior .02500), so the
outcome is not confined to one final-seed split.

At future horizon five, 16 final episodes are eligible and four die: observed
continuation **.75**. Complete-episode priors predict mean **.9380..9476**, an
absolute calibration gap **.1880..1976**. Predictions all remain in the high
continuation bin, so no death is identified at probability threshold .5.
Aggregate accuracy around .9615 and ECE .0147..0241 conceal this failure because
continuation dominates. Later horizons exclude already-ended episodes; counts
and labels accompany every horizon to expose this survivorship effect.

## Evidence and uncertainty

- **Small data/variance:** six training and four final death endpoints; three
  initialization seeds give the reward-error range above. Similar development
  and final patterns add evidence, but these are not confidence intervals or a
  broad benchmark. A small aggregate BCE difference is not evidence of control.
- **Posterior/prior mismatch:** complete-episode posterior reward MSE
  .02181..02710 versus prior .03822..05753. The report measures head gaps,
  memory MSE and posterior-to-open-loop-prior categorical KL by horizon. One seed
  has zero memory divergence but nonzero categorical/head gaps, so recurrent drift
  alone cannot explain every discrepancy. This is consistent with insufficiently
  constrained prior features and posterior-trained heads. It does not isolate the
  free-nat floor, mode approximation or partial observability as a sole cause.
  No new training objective was justified or substituted.
- **Code defects:** no observation/action/reward index or continuation-label defect
  was found. Fixtures cover final reward/continuation, death, exclusive truncation,
  simultaneous flags, endpoint copying and reset separation. Changing future
  observations cannot change prior reward/continuation predictions but changes
  the posterior fork. A numeric Brier/BCE/calibration oracle and complete-batch
  endpoint/copying tests were added. Tests do not prove absence of all defects.

No updates or hyperparameters were changed after final scores. No planner, actor,
learned control or policy evaluation was added. Before relying on imagination or
search, establish informative death coverage and prior predictive calibration
under a separately fixed development protocol. There is no survival-improvement
or continual-learning claim.

## Reproduction

From sibling `survival-agent-pilot-source` with model/environment checkouts:

```sh
PYTHONPATH=../survival-env:../survival-agent OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 60s python -B -m examples.hybrid_stage_b_diagnostic --output runs/hybrid-stage-b-diagnostic
```

Use a fresh output directory. No package installs, downloads, GPU or paid services.
The installed legacy Torch/NumPy runtime retains its documented security limits.
Actual browser inference/QA, learned control and cross-version numerical
reproducibility remain unverified. The original baseline report is unchanged.
