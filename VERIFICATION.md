# Verification

Independent checkout: 5 tests passed using the available legacy runtime. The README example command(s) ran successfully; baseline output was redirected to a fresh temporary directory. Isolation tests raise on imports from the other repository. No dependency installation, training experiment, model download, GPU, engine or paid resource was used.

Python 3.10.12 / NumPy 1.26.4. Agent checks additionally used Torch 2.2.0+cu121 and safetensors 0.4.5 on CPU. These old ML versions have known advisories. Recommended newer dependency resolution and installation are untested. No security certification is claimed. README dependency-install descriptions are prerequisites, not executed installation tests.

Public reciprocal links cannot be verified until the repositories have been created. Source extraction came from survival-agent-pilot commit be2e3f43b972a6cd83b0f31f420abdfd3b3a7f1e. Prior development history is retained there. The source-only extraction excludes models, checkpoints and private artifacts.
