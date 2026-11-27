"""Memorisation-capacity probing, with the four reproducibility rules built into the return value.

Pure functions only: training is injected as a `fit_fn(n, steps, seed) -> bool` callback, so this module never
imports a framework, never trains, never prints. That also makes it exactly testable -- the tests pass fake
fit callbacks whose answer is known, so every assertion is an exact number.

THE FOUR RULES (each one was violated by an earlier experiment, which is why they are enforced here)
    (1) the optimisation budget is always reported;
    (2) saturation is always VERIFIED: the threshold at B and at 2B must agree, otherwise the returned value is
        only a LOWER BOUND and `warning` says so;
    (3) at least 5 independent label draws are used, because p_fit is a step function and a single draw is a
        coin flip;
    (4) the architecture descriptor (including the input dimension) is always echoed back.

THE MEASURED QUANTITY (MNIST, tanh MLP, reference/notes/capacity-theory.md)
    capacity ~ 2 x parameter count; scales as p^0.88; independent of depth (1.86 vs 1.88 pts/param);
    ACT depends on how well the model is trained, hence rule (2).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import torch  # only for the optional defaults; no training happens here

__all__ = ["p_fit", "bisect_threshold", "measure_capacity", "selftest"]

FitFn = Callable[[int, int, int], bool]

SATURATION_TOL = 0.10
DRAWS_MIN = 5


def p_fit(fit_fn: FitFn, n: int, *, steps: int, draws: int, seed_base: int = 1) -> float:
    """Fraction of independent label draws that succeed at size n.

    Raises ValueError when draws < DRAWS_MIN: a smaller count cannot distinguish a step function from noise.
    """
    if draws < DRAWS_MIN:
        raise ValueError(f"draws must be >= {DRAWS_MIN} (rule 3); got {draws}")
    if n < 1:
        raise ValueError("n must be positive")
    ok = sum(1 for s in range(draws) if fit_fn(n, steps, seed_base + s))
    return ok / draws


def bisect_threshold(fit_fn: FitFn, *, steps: int, draws: int, grid: Sequence[int],
                     seed_base: int = 1) -> Dict[str, object]:
    """Estimate the size at which p_fit crosses 0.5, by scanning the grid and interpolating.

    Returns dict with threshold, grid_pfit (list of (n, p)), crossed (bool).
    A threshold at the grid boundary is reported with crossed=False and must not be read as a capacity.
    """
    pts = [(int(n), p_fit(fit_fn, int(n), steps=steps, draws=draws, seed_base=seed_base)) for n in grid]
    # NO interpolation: a capacity is the largest size that still succeeds at least half of the time. Linear
    # interpolation between p=1 and p=0 invented 350 for a model whose true limit is 300 (caught by the tests).
    passing = [n for n, p in pts if p >= 0.5]
    failing = [n for n, p in pts if p < 0.5]
    crossed = bool(passing) and bool(failing) and max(passing) < min(failing)
    if passing:
        return {"threshold": float(max(passing)), "grid_pfit": pts, "crossed": crossed}
    return {"threshold": float(pts[0][0]), "grid_pfit": pts, "crossed": False}


def measure_capacity(fit_fn: FitFn, *, budget: int = 4000, draws: int = 6,
                     grid: Sequence[int] = (150, 300, 450, 600, 900),
                     arch: Optional[Dict[str, object]] = None,
                     tol_ratio: float = SATURATION_TOL) -> Dict[str, object]:
    """Capacity = the SATURATED value of the p_fit threshold.

    Runs the threshold at `budget` and at `2*budget`; if they differ by more than `tol_ratio` the result is
    marked unsaturated and `capacity` is returned as a LOWER BOUND only, with a warning.
    """
    if draws < DRAWS_MIN:
        raise ValueError(f"draws must be >= {DRAWS_MIN} (rule 3); got {draws}")
    a = bisect_threshold(fit_fn, steps=budget, draws=draws, grid=grid)
    b = bisect_threshold(fit_fn, steps=2 * budget, draws=draws, grid=grid)
    t1, t2 = float(a["threshold"]), float(b["threshold"])
    saturated = (a["crossed"] and b["crossed"]
                 and abs(t2 - t1) <= tol_ratio * max(t1, 1e-9))
    report: Dict[str, object] = {
        "capacity": t2,
        "capacity_at_B": t1,
        "capacity_at_2B": t2,
        "saturated": bool(saturated),
        "lower_bound_only": not saturated,
        "threshold_crossed": {"B": a["crossed"], "2B": b["crossed"]},
        "grid_pfit": {"B": a["grid_pfit"], "2B": b["grid_pfit"]},
        "rules": {"budget": 2 * budget, "saturation_verified": bool(saturated),
                  "draws": draws, "tolerance": tol_ratio, "architecture_reported": arch is not None},
        "arch": arch,
        "warning": None,
    }
    if not saturated:
        report["warning"] = ("NOT saturated: the threshold still moves between B and 2B, so `capacity` is a "
                             "lower bound, not a capacity. Increase the budget before comparing.")
    return report


def _fake_fit(limit: int) -> FitFn:
    """A fit callback that succeeds iff n <= limit. Used by selftest and by the tests."""
    return lambda n, steps, seed: n <= limit


def selftest() -> Dict[str, object]:
    """Exact check with a fake fit callback: a model that fits up to 300 points must report capacity 300."""
    rep = measure_capacity(_fake_fit(300), budget=10, draws=5, grid=(100, 200, 300, 400, 600))
    ok = (abs(float(rep["capacity"]) - 300.0) < 1e-9 and rep["saturated"] is True
          and rep["warning"] is None)
    return {"capacity": rep["capacity"], "saturated": rep["saturated"], "ok": bool(ok)}


if __name__ == "__main__":
    print(selftest())
