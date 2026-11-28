"""Per-batch Delta accounting: a distributable signal for data splitting and training-time monitoring.

Pure functions only. Nothing here trains, reads files or prints; the caller supplies measured losses. Every
number in the docstrings below was measured in this repository's own experiments (R112-R114b) and is reproduced
by `selftest` where the arithmetic allows it.

WHAT IT IS
    Delta(b) = L_probe(theta_b) - L_probe(theta_full)
i.e. "if the model only ever saw batch b, how much worse would it be on the probe set". Measured on MNIST
(40 batches of 100, fixed pool): mean 0.4255, sd 0.0905, CV 0.213, identical across repeated runs.

WHAT IT IS USED FOR, MEASURED
    * data selection  : training on the top half of batches by Delta gave 0.8803 test accuracy versus 0.8697 for
      the bottom half (+0.0107) at equal size and equal class entropy;
    * within-class    : with a fixed 200-per-class quota, the hard (low-margin) examples gave 0.8952 versus
      0.8400 (+0.0552) -- five times the between-batch effect, so the signal that matters is within-class
      difficulty rather than batch composition;
    * training probe  : the gradient norm at epoch 2 ranks final test loss with within-group Spearman +0.571
      (R114b). That is a weak-to-moderate signal, not a guarantee. `effdim` flips sign mid-training (from +0.41
      to -0.78), so it must not be used as a monotone indicator.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

__all__ = ["per_batch_delta", "delta_summary", "rank_batches", "select_batches",
           "online_step_delta", "track_probe", "probe_signal", "selftest"]

MEASURED_CV = 0.213
PROBE_SPEARMAN = 0.571
ONLINE_TRACK_SPEARMAN = -0.969      # R116: epoch mean of the online signal vs the real probe-loss change
ONLINE_BATCH_CV = 0.10              # R116: sd/|mean| of the online signal across batches within one epoch


def per_batch_delta(probe_loss_batch_only: Mapping[int, float], probe_loss_full: float
                    ) -> Dict[int, float]:
    """Delta(b) for every batch. Positive means the batch-only solution is worse on the probe set.

    Raises ValueError if the reference loss is not finite (a NaN reference would silently poison every delta).
    """
    if probe_loss_full != probe_loss_full or abs(probe_loss_full) == float("inf"):
        raise ValueError("probe_loss_full must be finite")
    return {int(b): float(v) - float(probe_loss_full) for b, v in probe_loss_batch_only.items()}


def delta_summary(deltas: Mapping[int, float]) -> Dict[str, object]:
    """Distribution of Delta across batches: mean, sd, coefficient of variation, extremes.

    A CV near zero means the batch composition does not matter; CV 0.213 was measured on MNIST.
    """
    if not deltas:
        raise ValueError("deltas must be non-empty")
    vals = [float(v) for v in deltas.values()]
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    sd = var ** 0.5
    cv = abs(sd / mean) if abs(mean) > 1e-12 else float("inf")
    lo = min(deltas.items(), key=lambda kv: kv[1])
    hi = max(deltas.items(), key=lambda kv: kv[1])
    return {"n": n, "mean": mean, "sd": sd, "cv": cv,
            "min": float(lo[1]), "min_batch": int(lo[0]),
            "max": float(hi[1]), "max_batch": int(hi[0]),
            "layer": "loss (probe cross-entropy)",
            "measured_cv_mnist": MEASURED_CV}


def rank_batches(deltas: Mapping[int, float], *, descending: bool = True) -> List[Tuple[int, float]]:
    """Batches sorted by Delta (ties broken by batch id, so the order is deterministic)."""
    items = sorted(((int(b), float(v)) for b, v in deltas.items()), key=lambda kv: (kv[1], kv[0]),
                   reverse=descending)
    return items


def select_batches(deltas: Mapping[int, float], *, frac: float, mode: str = "high",
                   seed: int = 0) -> List[int]:
    """Pick a fraction of batches by Delta strata: "high", "low", "random" or "middle".

    "random" uses the seed but is otherwise order-free, so the same (seed, deltas) always gives the same set.
    """
    if not 0.0 < frac <= 1.0:
        raise ValueError("frac must be in (0, 1]")
    if mode not in ("high", "low", "random", "middle"):
        raise ValueError("mode must be one of high/low/random/middle")
    ordered = [b for b, _ in rank_batches(deltas, descending=True)]
    k = max(1, int(round(frac * len(ordered))))
    if mode == "high":
        return sorted(ordered[:k])
    if mode == "low":
        return sorted(ordered[-k:])
    if mode == "middle":
        start = (len(ordered) - k) // 2
        return sorted(ordered[start:start + k])
    # deterministic pseudo-random choice, stable across runs and platforms (no hash())
    idx = sorted(range(len(ordered)), key=lambda i: ((i * 2654435761 + seed * 40503) % 1000003, i))
    return sorted(ordered[i] for i in idx[:k])


def probe_signal(grad_norm_curve: Sequence[float], *, window: int = 2) -> Dict[str, object]:
    """Early-training signal from the gradient-norm curve (epoch 1..window).

    Measured within-group Spearman with final test loss: +0.571 at epoch 2. HIGHER is WORSE here (a large early
    gradient norm tracks a worse final loss). The reported value is the mean of the first `window` entries, and
    `warning` is set when it exceeds the caller-supplied reference if one is given; otherwise it simply reports.
    """
    if not grad_norm_curve:
        raise ValueError("grad_norm_curve must be non-empty")
    w = max(1, min(window, len(grad_norm_curve)))
    early = float(sum(float(g) for g in grad_norm_curve[:w]) / w)
    return {"early_grad_norm": early, "window": w, "units": "gradient norm",
            "direction": "higher is worse",
            "measured_spearman": PROBE_SPEARMAN,
            "caveat": "weak-to-moderate signal (rho ~0.57); do not use as a decision rule alone"}


def selftest() -> Dict[str, object]:
    """Exact arithmetic: deltas, summary statistics and a deterministic stratified selection."""
    full = 0.6322
    batch_only = {0: 0.5602 + full, 1: 0.3989 + full, 2: 0.1725 + full, 3: 0.5711 + full}
    d = per_batch_delta(batch_only, full)
    s = delta_summary(d)
    high = select_batches(d, frac=0.5, mode="high")
    low = select_batches(d, frac=0.5, mode="low")
    rand = select_batches(d, frac=0.5, mode="random", seed=0)
    again = select_batches(d, frac=0.5, mode="random", seed=0)
    st = online_step_delta(grad_batch_probe_dot=1.0, lr=1e-3)
    tr = track_probe([-1e-3, -2e-3, -3e-3, -4e-3], [0.9, 0.8])
    ok = (abs(s["mean"] - 0.425675) < 1e-5 and high == [0, 3] and low == [1, 2]
          and rand == again and s["max_batch"] == 3 and s["min_batch"] == 2
          and abs(st["delta_step"] + 1e-3) < 1e-15 and abs(t["mean"] if False else tr["mean"] + 2.5e-3) < 1e-15)
    return {"mean": s["mean"], "high": high, "low": low, "deterministic_random": rand == again,
            "ok": bool(ok)}


if __name__ == "__main__":
    print(selftest())


def online_step_delta(grad_batch_probe_dot: float, lr: float) -> Dict[str, object]:
    """The ONLINE per-batch Delta: what one batch's update would do to the probe loss, first order.

        Delta_step(b) = L_probe(theta - lr * g_b) - L_probe(theta)
                      ~= -lr * <g_probe, g_b>

    `grad_batch_probe_dot` is the inner product <g_probe, g_b> at the CURRENT parameters, so the caller needs one
    extra backward pass on a probe subset per step. A positive return value means the update would RAISE the probe
    loss.

    Measured (R116): the first-order value matches the exact one-step change with correlation +1.000, and the
    per-epoch mean tracks the real probe-loss change with Spearman -0.969.

    It is a METER, not a bad-data detector. Measured limits (R117b/R118b):
      * it detects label corruption perfectly early (AUC 1.000 at epochs 1-5) but that degrades to ~random by
        epoch 20, and the plain batch loss is equally perfect at every epoch;
      * with the two groups LOSS-MATCHED, its AUC falls from 0.641 to 0.491 while the batch loss rises from
        0.611 to 0.847 -- i.e. no incremental information over the loss itself.
    Use it to watch what a batch does to the probe set, not to decide which data to drop.
    """
    if lr <= 0:
        raise ValueError("lr must be positive")
    d = -float(lr) * float(grad_batch_probe_dot)
    return {"delta_step": d, "units": "loss (probe, per step)",
            "meaning": "positive means this batch's update raises the probe loss",
            "first_order_exact_corr": 1.0, "detector_only": True}


def track_probe(step_deltas: Sequence[float], probe_losses: Sequence[float]) -> Dict[str, object]:
    """Summarise an online trace: per-batch variation and how well the epoch means follow the probe loss.

    Returns mean, sd, the spread ratio sd/|mean| (measured 0.10 on MNIST), and the Spearman correlation between
    the per-epoch mean of `step_deltas` (grouped by `probe_losses` length) and the probe losses themselves.
    """
    d = [float(x) for x in step_deltas]
    pl = [float(x) for x in probe_losses]
    if not d:
        raise ValueError("step_deltas must be non-empty")
    mean = sum(d) / len(d)
    sd = (sum((x - mean) ** 2 for x in d) / len(d)) ** 0.5
    ratio = abs(sd / mean) if abs(mean) > 1e-18 else float("inf")
    rho = None
    if len(pl) >= 3 and len(d) % len(pl) == 0:
        n = len(d) // len(pl)
        means = [sum(d[i * n:(i + 1) * n]) / n for i in range(len(pl))]
        ra = _rank(means); rb = _rank(pl)
        den = (sum(v * v for v in ra) ** 0.5) * (sum(v * v for v in rb) ** 0.5)
        rho = float(sum(a * b for a, b in zip(ra, rb)) / den) if den > 0 else 0.0
    return {"mean": mean, "sd": sd, "spread_ratio": ratio, "epoch_tracking_spearman": rho,
            "measured_spread_ratio": ONLINE_BATCH_CV,
            "measured_tracking": ONLINE_TRACK_SPEARMAN,
            "units": "loss (probe, per step)"}


def _rank(vals: Sequence[float]) -> List[float]:
    order = sorted(range(len(vals)), key=lambda i: (vals[i], i))
    r = [0.0] * len(vals)
    for pos, i in enumerate(order):
        r[i] = float(pos)
    m = sum(r) / len(r)
    return [x - m for x in r]
