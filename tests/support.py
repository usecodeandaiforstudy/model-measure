"""Minimal helper so the suite runs with OR without pytest installed.

The verified GPU environment ships torch but no pytest, so the tests must not import it. `approx` mirrors
the small slice of `pytest.approx` that this suite uses (absolute OR relative tolerance), and the modules
stay plain `test_*` functions with asserts, which pytest collects natively.
"""

import builtins


class _Approx:
    def __init__(self, expected, rel=1e-6, abs=1e-12):
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def _tol(self):
        return max(self.abs, self.rel * builtins.abs(self.expected))

    def __eq__(self, other):
        try:
            diff = builtins.abs(float(other) - float(self.expected))
        except (TypeError, ValueError):
            return NotImplemented
        return diff <= self._tol()

    def __repr__(self):                                   # pragma: no cover - diagnostics only
        return f"approx({self.expected}, tol={self._tol():g})"


def approx(expected, rel=1e-6, abs=1e-12):
    """Compare with a tolerance: passes when |other - expected| <= max(abs, rel*|expected|)."""
    return _Approx(expected, rel=rel, abs=abs)
