"""Compatibility path for survival_adapters.language_run."""
import sys
from survival_adapters import language_run as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
