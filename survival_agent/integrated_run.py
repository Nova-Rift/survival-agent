"""Compatibility path for survival_adapters.integrated_run."""
import sys
from survival_adapters import integrated_run as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
