# Security scope

This public repository contains curated source, tests and documentation. No model weights, credentials, private run data or checkpoints are included. Functional tests are not a security certification.

The original bounded review, serialization fixes and advisory context remain documented in [the integration project's security scope](https://github.com/Nova-Rift/survival-agent-pilot/blob/main/SECURITY.md). The model package uses bounded structured JSON/safetensors rather than pickle; the environment package does not load model files. Replay JSON escaping is retained. No project script installs dependencies automatically.

Legacy functional-test dependencies have known advisories. Recommended newer ML requirements are not fully compatibility-tested together. No fresh dependency installation, wheel build, exhaustive transitive scan, hostile-model sandbox or standards compliance is claimed. Use isolated current dependencies for real work. Do not publish secrets or private checkpoint contents in issues.
