"""Subset representativeness: is a data subset unlike the full task?

Pure functions only: gradient magnitudes are supplied by the caller (one float per subset), so this module
does arithmetic and decision logic, never training and never I/O. That makes every test an exact number.

WHAT THE SCORE CAN AND CANNOT DO (measured; see reference/notes/subspace-exploration-summary.md)
    CAN    flag a subset that looks UNLIKE the full task (between-kind AUC 1.000 in the runs performed);
    CANNOT select the best subset (picking the smallest rho' is no better than random: -0.0004);
    CANNOT be read as an accuracy prediction.
    Every report therefore carries `validity.window_steps` and `boundaries`: the signal's rank correlation with
    post-retraining loss is 0.900 within about 100 optimisation steps and decays to 0.600 by 1000 steps.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

__all__ = ["calibrate_threshold", "subset_rho_prime", "align_report", "selftest"]

WINDOW_STEPS = 100
TAU_FLOOR = 2.0
Q95_MULT = 1.5
MIN_BASELINES = 10


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted sequence. Deterministic, no numpy dependency."""
    if not sorted_vals:
        raise ValueError("empty sequence")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def calibrate_threshold(baseline: Iterable[float], *, mult: float = Q95_MULT,
                        floor: float = TAU_FLOOR) -> Dict[str, object]:
    """tau = max(mult * q95(baseline), floor).

    The baseline must be aligned to the SAME baseline used for rho', i.e. representative subsets of the full
    task. Fewer than MIN_BASELINES points means the lower bound is not stable (measured CV 36% at 4 samples per
    arm vs 3.9% at a larger budget), so `stable` is returned False and the caller must not treat tau as final.
    """
    vals = sorted(float(v) for v in baseline)
    if not vals:
        raise ValueError("baseline must be non-empty")
    if any(v <= 0.0 for v in vals):
        raise ValueError("baseline values must be positive (they are gradient magnitudes)")
    q95 = _quantile(vals, 0.95)
    tau = max(mult * q95, floor)
    return {"tau": tau, "q95": q95, "floor": floor, "mult": mult,
            "n_baseline": len(vals), "stable": len(vals) >= MIN_BASELINES,
            "note": None if len(vals) >= MIN_BASELINES else
            f"fewer than {MIN_BASELINES} baseline subsets: the lower bound is not stable"}


def subset_rho_prime(aligns: Mapping[str, float], baseline: Iterable[float]) -> Dict[str, object]:
    """rho'(A) = align(A) / median(align of baseline).

    The median of the baseline is the normaliser, so the score is dimensionless and comparable across tasks.
    """
    base = sorted(float(v) for v in baseline)
    if not base:
        raise ValueError("baseline must be non-empty")
    med = _quantile(base, 0.5)
    if med <= 0.0:
        raise ValueError("median baseline must be positive")
    out = {k: float(v) / med for k, v in aligns.items()}
    return {"rho_prime": out, "normaliser": med, "n_baseline": len(base)}


def align_report(aligns: Mapping[str, float], baseline: Iterable[float], *,
                 window_steps: int = WINDOW_STEPS) -> Dict[str, object]:
    """Per-subset verdict plus the mandatory validity block.

    Returns dict with subsets (list of {name, rho_prime, verdict}), tau, tau_rule, validity, boundaries.

    `verdict` is "not-representative" when rho_prime >= tau and "representative" otherwise. It answers only
    "does this subset look unlike the full task"; it must never be read as "this subset is better".
    """
    cal = calibrate_threshold(baseline)
    rho = subset_rho_prime(aligns, baseline)
    tau = float(cal["tau"])
    subsets: List[Dict[str, object]] = []
    for name in sorted(rho["rho_prime"]):
        v = float(rho["rho_prime"][name])
        subsets.append({"name": name, "rho_prime": v,
                        "verdict": "not-representative" if v >= tau else "representative"})
    return {
        "subsets": subsets,
        "tau": tau,
        "tau_rule": "max(1.5 * q95(baseline), 2.0)",
        "calibration": cal,
        "validity": {
            "window_steps": window_steps,
            "what_it_predicts": "cost of using the subset, not the benefit of choosing it",
            "note": ("rank correlation with post-retraining loss is 0.900 up to about 100 steps and 0.600 by "
                     "1000 steps; always state the optimisation budget"),
        },
        "boundaries": [
            "no absolute threshold: the measured within-kind discrimination does not support a yes/no verdict",
            "the score needs the full dataset (it uses the full-data gradient)",
            "selecting the smallest rho' subsets is no better than random",
            "cannot predict a subset's accuracy",
        ],
    }


def selftest() -> Dict[str, object]:
    """Exact check: baselines of 1.0 give tau=2.0, and rho'=4.0 must be flagged."""
    rep = align_report({"good": 1.0, "bad": 4.0}, [1.0] * 12)
    got = {s["name"]: s["verdict"] for s in rep["subsets"]}
    ok = (abs(rep["tau"] - 2.0) < 1e-12 and got["good"] == "representative"
          and got["bad"] == "not-representative" and rep["validity"]["window_steps"] == 100)
    return {"tau": rep["tau"], "verdicts": got, "ok": bool(ok)}


if __name__ == "__main__":
    print(selftest())
