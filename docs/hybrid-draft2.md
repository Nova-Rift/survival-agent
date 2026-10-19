# Experimental hybrid: architecture and acceptance plan

October 19 update: at Clark's request, the current implementation uses
[causal transformer memory](../survival_agent/hybrid/transformer_agent.py), with
the prior/posterior, stochastic state and prediction roles retained. See the
[README component map](../README.md) for the active architecture. The original
GRU implementation and its experimental results are preserved. The historical
stage table below predates the real-interaction actor/critic and this change;
imagination learning and the proposed planners remain unimplemented.

[Stage A implementation and commands](hybrid-stage-a.md) · [Stage B world model](hybrid-stage-b.md) · [Pinned references](reference-pins.json) · [Existing baseline](../README.md)

This design is an experimental combination of DreamerV3-inspired recurrent stochastic world modeling/imagination, MuZero-inspired discrete search and TD-MPC2-inspired continuous control. It is not a faithful implementation of all three, and no performance advantage is established. Existing history/latent models remain the old baseline; they are not relabeled Draft-2 compliant.

## Architecture delta

| Component | Required new design | Current status |
|---|---|---|
| Tasks | Single-agent vector CartPole, Pendulum and deterministic delayed reward; one task per run | Stage-A collection/data smoke only |
| Actions | Declared discrete count or finite continuous bounds; normalized continuous model/replay coordinates | Stage A |
| World model | Encoder, deterministic recurrent memory plus stochastic state, observation-conditioned posterior, prior dynamics, decoder, reward and continuation heads | Stage B small vector RSSM implemented; CPU smoke only |
| World-model training | Real sequence reconstruction/reward/continuation and balanced prior/posterior losses; pinned stabilization details | Stage B implemented with documented simplifications; held-out multi-step reward/continuation errors remain concerning |
| Actor / state value | Posterior-seeded imagination, lambda returns, entropy/normalization controls and reference-specific gradient boundaries | Stage C, unimplemented |
| Continuous Q ensemble | Distinct real-replay TD critics, slowly updated targets and explicit ensemble reduction; initially no world-model gradients | Stage D, unimplemented |
| MPC | Shifted warm start, actor proposals and bounded weighted sampling; Q ensemble with actor action at horizon, state-value ablation | Stage D, unimplemented |
| MCTS | Actor priors, prior-only latent transitions, state-value leaves, discounted backups and root visit targets | Stage E, unimplemented |
| Distillation | Optional/off initially; detached discrete visits or continuous action targets anchored to real replay/history and model version | Stage F, unimplemented |
| Comparisons | Actor-only vs relevant planner, with/without distillation, frozen-checkpoint on/off, multiple seeds and interaction/compute budgets | Stage G, not run |

Automatic dispatch declares actor when planning is disabled, MCTS for discrete and MPC for continuous. Stage A returns these mode names only: it does not instantiate a planner or call the old prototype planner as a substitute.

The shared model must preserve both recurrence and stochastic state. Real interaction updates the posterior; imagination/search uses the prior without future real observations. A training decoder must not be required for each search transition. Dreamer state-value and continuous real-replay action-value critics stay distinct. The latter's stopped representation gradients deliberately differ from TD-MPC2 joint, decoder-free training.

## Index and discount convention

A transition is `(o_t, a_t, r_t, o_next, terminated_t, truncated_t)`. Reward `r_t` belongs to the action that produced `o_next`. Continuation is **`1 - terminated_t`**, with gamma applied separately exactly once. Never suppress the reward already received on a terminal transition. External truncation retains final-observation bootstrapping, while either end flag resets interaction memory and MPC warm starts.

Dreamer reference options can fold gamma into learned continuation (`contdisc`); that convention must not be copied on top of our separate gamma. Before stage C, inspect the pinned actor implementation and preserve its detached-advantage/log-probability and stop-gradient behavior; do not silently substitute a different Dreamer version's pathwise objective. Reference stabilization choices (KL balancing, free bits, distributions, transforms, return normalization) must be recorded with their actual tests, not inferred from this plan.

Numeric acceptance: gamma 0.5, rewards 1 then 2, final value 4 gives `1 + .5*2 + .25*4 = 3` through truncation. If the second reward truly terminates, the same sequence gives 2. The final value must never come from the next episode's reset observation.

## Planner invariants for later stages

- Search receives copied latent state and a fixed model/actor/critic snapshot. Only the collector owns real environment execution. Search must make zero step/reset/clone/state-set/hidden-state calls and leave live recurrence unchanged.
- Root legality is actual observed root information. Illegal root actions receive zero probability. Do not reuse the root mask at imagined descendants or call environment legality there. Start with all-actions-valid tasks; later choose unmasked interiors or a fallible learned legality model explicitly.
- Use one-player reward signs. Undo reward/value transforms before adding predictions in real reward units; continuation gates future returns, not received reward.
- Initial planning uses a declared deterministic latent approximation (for example categorical mode) with cached transitions. This is not general stochastic planning. Later MPC must average outcomes and MCTS needs chance-aware expected backups, not cherry-picked outcomes.
- Continuous MPC uses an explicitly documented TD-MPC2-style optimizer. CEM may be a named baseline, not silently substituted. Execute only the first bounded action, then incorporate the real observation and replan.
- Distillation targets are detached, age/version bounded and reconstructed from current replay histories. A preferred planner action is not automatically a better teacher.

## Acceptance gates

| Gate | Evidence needed before advancing |
|---|---|
| A | Exact observation/action/reward/flag alignment; normalized bounds; episode-safe replay/burn-in; true final observations; per-stream reset lifecycle; safe boundary checkpoint/RNG/sample/fixed-seed evaluation resume |
| B | Recurrent/posterior/prior/decoder/head shapes; causal indices; finite losses/gradients; held-out one/multi-step errors; documented stabilization simplifications |
| C | Actor-only learning demonstrated; posterior starts from real replay; correct lambda returns and gradient stops; actor updates do not change world-model parameters |
| D | Real-next-observation TD targets, termination/truncation and target ensemble checks; MPC bounds/warm starts/budgets; sensible toy choices; zero environment calls/live-state mutation |
| E | Root probabilities/masks/visit accounting; correct single-agent discounted backups; deterministic transition caching; sensible toy choices; no environment access |
| F | Detached, versioned/aged targets with observation-history references; separate continuous/discrete targets; no-distillation control |
| G | At least three initial training seeds; actual returns, sample efficiency, latency/model calls, time/memory/variation; separate matched-interaction and matched-compute comparisons; honest negative results |

Full completion requires actual actor-only learning, learned-model planner tests and controlled comparisons. Stage-A smoke tests alone do not meet that definition. Images, multiplayer, mixed spaces, hierarchy and real deployment remain deferred. No GPU training, paid compute or long run is authorized by the smoke commands.

## Reference and licensing status

The [reference manifest](reference-pins.json) pins official implementation revisions and verified upstream MIT license metadata; the paper references are pinned by identifier/version where available. Stage A is original glue/data code using the public Gymnasium API; no third-party algorithm source was copied. Inspect exact source and preserve attribution/license notices before borrowing code in later stages. Upstream MIT licenses do not select a license for this repository; project licensing remains pending.

The Nature paper page could not be retrieved by the browsing tool in the original reference pass. No claim is made that its complete text was read. Stage B inspected pinned `rssm.py` and `configs.yaml` for KL stops, free nats, uniform mixing, distribution types and loss scales; its reduced PyTorch implementation is not full DreamerV3.

## World roadmap remains the product goal

CartPole/Pendulum and the delayed-reward task validate agent mechanics; they do not replace survival research. Stage B's real-replay CPU smoke uses the lightweight single-agent survival world. The candidate editable 3D world is Luanti with VoxeLibre. Future integration must support equal-capability human-like actors with separate memories, local perception/nearby communication, wildlife, resources and weather as actually supported. Integration, Earth fidelity and learned human behavior are not established. Stage C and controlled survival/retention comparisons are still pending.
