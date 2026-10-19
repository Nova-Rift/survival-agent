# Survival Agent

The new [Draft-2 hybrid design](docs/hybrid-draft2.md) has a separate [stage-A data/replay/checkpoint implementation](docs/hybrid-stage-a.md). RSSM, imagination learning and the new planners remain unimplemented. The source below remains the original baseline.

Canonical experimental models, local language backend, bounded latent search and model-state serialization. The concrete grid environment is independently maintained in [survival-env](https://github.com/Nova-Rift/survival-env). World collection, adapters, combined checkpoints and examples live in [survival-agent-pilot](https://github.com/Nova-Rift/survival-agent-pilot).

## Quickstart from this checkout

Python 3.10+, NumPy, Torch and safetensors must already be available. No model download or Transformers is needed for these commands:

```bash
python -B -m examples.inspect_model
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m unittest discover -v
```

The example runs randomly initialized model heads on synthetic observations and checks finite logits/values. There are no optimizer updates, downloaded weights, APIs or useful trained results. Tests block all concrete-environment and integration imports.

## Dependencies and limits

`requirements.txt` and package metadata define NumPy, security-oriented Torch/safetensors minimums; `requirements-language.txt` and the `language` extra add Transformers/Jinja2. **Fresh dependency installation and wheel build are untested.** Functional checks used legacy Python 3.10.12, NumPy 1.26.4, Torch 2.2.0+cu121 and safetensors 0.4.5 on CPU. This legacy runtime has known advisories; passing checks are not security certification. No dependencies were installed or upgraded.

`HistoryAgent` consumes `(batch, history, 5, obs_width, obs_width)`, normally history 4 and width 5. `LanguageLatentCore` consumes bounded feature vectors. Existing policy heads use **five discrete actions**. `MuZeroModel`/`mcts` provide a minimal PUCT/replay/unroll research scaffold, not full MuZero. Importing core models/search never imports a concrete simulator. Legacy simulator and CLI modules are forwarding shims: explicitly invoking one requires the corresponding environment/integration checkout on the Python path. Combined checkpoint and collector helpers also delegate lazily to pilot adapters. Collectors and environment reconstruction belong in the pilot adapters, not here.

The optional language backend accepts vetted local directories and safetensors only; it does not implicitly download. Scratch language mode is random weights with a byte vocabulary. Supported pretrained integration keeps the encoder frozen; full pretrained fine-tuning is unsupported. Some language formatting remains survival-specific. This package does not promise arbitrary observation/action spaces, continuous controls or policy transfer.

## Checkpoints

`checkpoint_io` uses bounded structured JSON and safetensors, rejecting pickle/Torch legacy archives, unknown types, unsafe arrays and nonfinite metadata. Episode dimensions are checked before replay boundary arithmetic. Model-only state belongs here; combined model/world state belongs in pilot adapters. Source/runtime mismatch may intentionally reject resume. Never bypass provenance checks or publish private checkpoints. See [security scope](SECURITY.md).

## Ownership

Edit model/search/serialization code here. The pilot repo will pin this repository rather than maintain another copy. Source was extracted from pilot commit `be2e3f43b972a6cd83b0f31f420abdfd3b3a7f1e`; prior development history stays there. No project source license has been selected; public visibility is not an open-source license grant. Third-party pretrained artifacts retain their own licenses.
