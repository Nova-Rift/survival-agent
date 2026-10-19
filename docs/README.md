# Research and implementation notes

- [Current agent explained: component map, one-step flow and training](../README.md) — the transformer agent is implemented in one model file; continual learning and forgetting protection remain unfinished.
- [Readable transformer agent source](../survival_agent/hybrid/transformer_agent.py)
- [Continual learning / plasticity proposal, architecture inspection and smallest experiment](online-plasticity-review.md) — proposed, not implemented. Preserves replay/TD priorities, historical-skill safeguards, matched controls and adoption gates.
- [Nature: Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7) — retained primary reference; a motivation for testing, not a retention guarantee.
- [Draft-2 architecture and gates](hybrid-draft2.md)
- [Stage A: data, replay and checkpoints](hybrid-stage-a.md)
- [Stage B: recurrent world model](hybrid-stage-b.md)
- [Initial posterior/prior diagnostic](hybrid-stage-b-diagnostics.md)
- [Larger matched-budget data-coverage diagnostic](coverage-diagnostic.md)
