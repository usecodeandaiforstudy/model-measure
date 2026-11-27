"""Bit-side curves: how much loss a representation needs, and where pruning starts to hurt.

Pure functions only: the caller supplies precomputed curves (bits -> loss, units pruned -> loss), so this module
is arithmetic and nothing else. Every test is therefore an exact number.

MEASURED SHAPES (reference/notes/delta-information-theory.md, reference/notes/structure-space-delta.md)
    activation quantisation   saturates at ~4 bits per dimension (Delta = 0.0809, 0.0221, 0.0076, -0.0009)
    weight quantisation       saturates at ~6 bits per parameter
    structured pruning        k=1..3 of 6 units costs +0.003..+0.021, k=4 jumps to +0.264  <- a cliff
    the price per bit differs ~9x between the sides, so NO cross-side conversion is provided on purpose.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

__all__ = ["activation_curve", "pruning_curve", "selftest"]

DEFAULT_TOL = 1e-2
DEFAULT_CLIFF_RATIO = 5.0
DEFAULT_MIN_COST = 1e-3


def activation_curve(losses: Mapping[int, float], baseline: float, *,
                     tol: float = DEFAULT_TOL) -> Dict[str, object]:
    """Compute Delta(b) = loss(b) - baseline and locate the saturation bit width.

    `saturated_at` is the smallest b such that every b' >= b satisfies |Delta| <= tol, so it is stable under
    non-monotone noise. Returns delta, saturated_at, monotone, units (loss).
    """
    if not losses:
        raise ValueError("losses must be non-empty")
    if any(b < 0 for b in losses):
        raise ValueError("bit widths must be non-negative")
    ks = sorted(int(b) for b in losses)
    delta = {b: float(losses[b]) - float(baseline) for b in ks}
    mono = all(delta[ks[i]] >= delta[ks[i + 1]] - 1e-12 for i in range(len(ks) - 1))
    sat: Optional[int] = None
    for i, b in enumerate(ks):
        if all(abs(delta[kk]) <= tol for kk in ks[i:]):
            sat = b
            break
    return {"delta": delta, "saturated_at": sat, "monotone": bool(mono), "tol": tol,
            "units": "loss", "baseline": float(baseline)}


def pruning_curve(costs: Mapping[int, float], baseline: float, *,
                  cliff_ratio: float = DEFAULT_CLIFF_RATIO,
                  min_cost: float = DEFAULT_MIN_COST) -> Dict[str, object]:
    """Locate the pruning cliff: the first k whose Delta grows by at least `cliff_ratio` over the previous k.

    `safe_k` is the largest k before the cliff, i.e. the deepest pruning that does not cross it.
    Returns delta, cliff_k, safe_k, monotone, units (loss).
    """
    if not costs:
        raise ValueError("costs must be non-empty")
    ks = sorted(int(k) for k in costs)
    if ks[0] < 0:
        raise ValueError("pruning counts must be non-negative")
    delta = {k: float(costs[k]) - float(baseline) for k in ks}
    mono = all(delta[ks[i]] <= delta[ks[i + 1]] + 1e-12 for i in range(len(ks) - 1))
    cliff: Optional[int] = None
    for i in range(1, len(ks)):
        prev = max(delta[ks[i - 1]], min_cost)
        if delta[ks[i]] >= cliff_ratio * prev and delta[ks[i]] > min_cost:
            cliff = ks[i]
            break
    safe = None
    if cliff is not None:
        smaller = [k for k in ks if k < cliff]
        safe = max(smaller) if smaller else None
    return {"delta": delta, "cliff_k": cliff, "safe_k": safe, "monotone": bool(mono),
            "cliff_ratio": cliff_ratio, "units": "loss"}


def selftest() -> Dict[str, object]:
    """Exact check on the measured activation curve and on a synthetic cliff."""
    act = activation_curve({1: 0.0809, 2: 0.0221, 3: 0.0076, 4: -0.0009, 8: -0.0001}, 0.0)
    pr = pruning_curve({1: 0.003, 2: 0.005, 3: 0.021, 4: 0.264, 5: 0.661}, 0.0)
    # the real activation curve is NOT monotone at b >= 4 (a -0.0009 dip then -0.0001), which is exactly why
    # `saturated_at` is defined as "every larger b also passes" instead of relying on monotonicity
    ok = (act["saturated_at"] == 3 and act["monotone"] is False
          and pr["cliff_k"] == 4 and pr["safe_k"] == 3)
    return {"saturated_at": act["saturated_at"], "monotone": act["monotone"],
            "cliff_k": pr["cliff_k"], "safe_k": pr["safe_k"], "ok": bool(ok)}


if __name__ == "__main__":
    print(selftest())
