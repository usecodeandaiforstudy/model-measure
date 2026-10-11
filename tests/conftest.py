"""Path bootstrap for the test suite.

The library is importable after `pip install -e .`; the tests also run straight from a checkout, so both
the repository root (the package) and `experiments/` (whose modules hold the pre-registered criterion
functions that this suite guards) are put on sys.path.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
