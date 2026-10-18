"""Compatibility path for survival_adapters.training."""
import sys
from survival_adapters import training as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
