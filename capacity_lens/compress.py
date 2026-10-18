"""Compression budget advisor: provable guards, measured allocation, honest intervals.

PRE-REGISTRATION (frozen 2026-10-15, written before the module was used on any new instance).

Positioning
-----------
This module is an ADVISOR, not a compressor. It never trains a compressed model and never claims a
performance gain. Every value it returns carries an evidence grade, and the grade is part of the
contract:

    [CERT]  provable from a theorem (Fano's inequality, capacity accounting)
    [MEAS]  measured on the instance at hand, with zero training of the thing being predicted
    [ARCH]  a constant measured earlier in the campaign on other instances, quoted with its task,
            representation and protocol
    [LOOSE] an estimate with a known looseness factor

What is certified
-----------------
* ``task_info_floor(acc, n_classes, h_y)`` -- R_IB, the information a C-class task at that error rate
  provably needs. Pass the EMPIRICAL label entropy whenever the classes are not balanced; with the
  default ``h_y = ln C`` the floor is overstated on unbalanced tasks.
* ``capacity_floors(rib)`` -- the codebook / bit / capacity lower bounds implied by R_IB.
* ``veto(k_codes, capacity_nats, rib)`` -- IMPOSSIBLE when the compressed information path is below
  the floor. Sound and necessary-only: the floor sits 10-30x below the practical knee on record, so
  the veto fires only on extreme over-compression.
* ``tightness_interval(i_lb, rib, h_y)`` -- the true task-relevant looseness is bracketed by
  ``[i_lb / R_IB, H(y) / R_IB]``: Fano gives the lower end a one-sided guarantee, and ``I <= H(y)``
  gives the upper end. On the campaign's own instances this brackets the certificate within 1.26x
  (MNIST) and 1.92x (CIFAR-10).

What is measured, and how loose it is
-------------------------------------
* ``partition_retention`` -- the fraction of I(z;y) that survives hard quantisation to K codes. On
  record: 0.65 (MNIST penultimate), 0.59 (CIFAR-10 conv trunk), 0.15-0.25 (a hard synthetic MLP),
  i.e. 0.15-0.75 and always below 1. Retention depends on the task AND on which representation is
  partitioned, so a report must name the representation.
* ``partition_ceiling`` -- the Fano ceiling implied by the retained information. Sound in direction,
  loose by 2-3x; use it as a veto, never as a prediction.

Boundaries this module refuses to hide
---------------------------------------
* A per-layer proxy must never pin a layer to its minimum: rank starvation is irreversible (measured
  -0.205 retention, 0 of 6 seeds recovered by fine-tuning). Bit starvation, by contrast, is
  recoverable by QAT, so mixed precision is allowed and helps most at tight average widths.
* Allocate only by whole-network evaluation on a validation split, never by a per-layer curve sum and
  never on the test set.
* Every wall is protocol-relative (learning rate and schedule move it by at least 8x on record), so a
  number is quoted together with its fingerprint.
* The toolkit's readings support guards and diagnoses; they do not make a model better.
"""
from __future__ import annotations

import math

try:
    import torch
    import torch.nn.functional as F
except Exception:                                        # noqa: BLE001  (pure parts must import)
    torch = None
    F = None

from .capacity import discrete_capacity_logK
from .distill import DistillCapacity
from .quant import QuantCapacity
from .training import mutual_info_of, task_info_fano

__all__ = [
    "BALANCE_TOL", "FLOOR_VS_KNEE", "RETENTION_RANGE", "CEILING_LOOSENESS",
    "balance_warning", "task_info_floor", "capacity_floors", "veto", "tightness_interval",
    "eps_for_rib", "partition_retention", "partition_ceiling", "kmeans", "selftest",
]

BALANCE_TOL = 0.01          # nats; beyond this the empirical h_y must be supplied
FLOOR_VS_KNEE = (10.0, 30.0)   # [ARCH] the certified floor sits this far below the practical knee
RETENTION_RANGE = (0.15, 0.75)  # [ARCH] hard quantisation retention at K=64, always < 1
CEILING_LOOSENESS = (2.0, 3.0)  # [ARCH] the partition ceiling is loose by this factor


def balance_warning(h_y, n_classes, tol=BALANCE_TOL):
    """[CERT] Warn when the balanced-class default h_y = ln C is not valid for this task."""
    gap = math.log(n_classes) - h_y
    if gap <= tol:
        return None
    return (f"labels are not balanced: H(y)={h_y:.4f} vs ln C={math.log(n_classes):.4f} "
            f"(gap {gap:.4f} nats). Compute R_IB with the empirical h_y: the default ln C "
            f"overstates the floor and therefore understates looseness.")


def task_info_floor(acc, n_classes, h_y=None):
    """[CERT] R_IB for an achieved accuracy (Fano, nats)."""
    return float(task_info_fano(1.0 - acc, n_classes, h_y))


def capacity_floors(rib, dims=(32, 64, 128)):
    """[CERT] Codebook / bit / capacity lower bounds implied by R_IB (nats in, nats and bits out)."""
    q = QuantCapacity()
    out = {"min_codebook_size": int(DistillCapacity().min_codebook_size(rib)),
           "min_capacity_nats": float(DistillCapacity().min_capacity_nats(rib)),
           "log_k_from_capacity": float(discrete_capacity_logK(max(2, int(math.ceil(math.e ** rib))))),
           "quant": {}}
    for d in dims:
        out["quant"][d] = {"min_bits_per_dim": float(q.min_bits_per_dim(rib / math.log(2.0), d)),
                           "min_levels": float(q.min_levels(rib / math.log(2.0), d))}
    return out


def veto(k_codes=None, capacity_nats=None, rib=None):
    """[CERT] IMPOSSIBLE when the compressed information path cannot carry the task.

    Sound and necessary-only: passing does NOT mean the compression will work, because the floor sits
    10-30x below the practical knee that behaviour actually needs.
    """
    if rib is None:
        return "UNDECIDABLE", "no floor given"
    cap = capacity_nats if capacity_nats is not None else (math.log(k_codes) if k_codes else None)
    if cap is None:
        return "UNDECIDABLE", "no bottleneck capacity (pass k_codes or capacity_nats)"
    if cap < rib:
        return "IMPOSSIBLE", f"capacity {cap:.3f} < R_IB {rib:.3f} nats"
    return "FEASIBLE", (f"capacity {cap:.3f} >= R_IB {rib:.3f}; necessary only -- the floor is "
                        f"{FLOOR_VS_KNEE[0]:.0f}-{FLOOR_VS_KNEE[1]:.0f}x below the practical knee, "
                        f"so do not configure to it")


def tightness_interval(i_lb, rib, h_y):
    """[CERT] Bracket the true task-relevant looseness: [i_lb / R_IB, H(y) / R_IB].

    The lower end uses Fano (I_true >= R_IB) and the upper end the trivial I_true <= H(y), so the
    interval is one-sided in the direction that matters: the certificate cannot be looser than the
    upper end.
    """
    if rib is None or rib <= 0:
        return None
    lo = max(1.0, i_lb / rib) if i_lb else 1.0
    return (lo, h_y / rib)


def eps_for_rib(target, n_classes, h_y=None):
    """[CERT] Invert Fano: the smallest error rate whose floor is at least ``target`` nats.

    R_IB decreases in eps, so the bisection brackets the answer from above.
    """
    lo, hi = 0.0, 1.0 - 1.0 / n_classes
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if float(task_info_fano(mid, n_classes, h_y)) > target:
            lo = mid
        else:
            hi = mid
    return hi


def kmeans(x, k, iters=25, seed=0):
    """Plain k-means; deterministic given the seed. Returns the centroids (k, d)."""
    if torch is None:
        raise RuntimeError("kmeans needs torch")
    g = torch.Generator().manual_seed(seed)
    ctr = x[torch.randperm(x.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(x, ctr).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                ctr[j] = x[m].mean(0)
    return ctr


def partition_retention(z_fit, y_fit, z_eval, y_eval, k, n_classes, seed=0):
    """[MEAS] Fraction of the task-relevant information that survives quantisation to K codes.

    Fit the codebook on ``z_fit`` and measure on a DISJOINT ``z_eval``: the plug-in mutual information
    collapses toward zero once the evaluation sample is not much larger than K * n_classes, so keep
    ``len(z_eval) >= 4 * k * n_classes``. Returns (retention, I_code, I_rep) where I_rep is the
    variational lower bound H(y) - CE_probe on the same split.
    """
    if torch is None:
        raise RuntimeError("partition_retention needs torch")
    if z_eval.shape[0] < 4 * k * n_classes:
        raise ValueError(f"need len(z_eval) >= 4*K*C = {4 * k * n_classes}, got {z_eval.shape[0]}")
    ctr = kmeans(z_fit, k, seed=seed)
    idx_fit = torch.cdist(z_fit, ctr).argmin(1)
    idx_eval = torch.cdist(z_eval, ctr).argmin(1)
    i_code = float(mutual_info_of(idx_eval, y_eval, k, n_classes))
    i_rep = probe_lower_bound(z_fit, y_fit, z_eval, y_eval, n_classes, seed=seed)
    return (i_code / i_rep if i_rep > 0 else float("nan")), i_code, i_rep


def probe_lower_bound(z_fit, y_fit, z_eval, y_eval, n_classes, seed=0, epochs=30, width=128):
    """[MEAS] Variational lower bound on I(z;y): H(y) - CE of a probe that sees only z."""
    if torch is None:
        raise RuntimeError("probe_lower_bound needs torch")
    import torch.nn as nn
    d = z_fit.shape[1]
    torch.manual_seed(seed)
    # The probe must live on the SAME device as the representation: a GPU caller otherwise fails with
    # "Expected all tensors to be on the same device" (found by the user walkthrough on 2026-10-15).
    probe = nn.Sequential(nn.Linear(d, width), nn.ReLU(), nn.Linear(width, n_classes)).to(z_fit.device)
    opt = torch.optim.Adam(probe.parameters(), lr=3e-3, weight_decay=1e-4)
    n = z_fit.shape[0]
    for _ in range(epochs):
        perm = torch.randn(n).argsort()
        for i in range(0, n, 1024):
            b = perm[i:i + 1024]
            opt.zero_grad()
            F.cross_entropy(probe(z_fit[b]), y_fit[b]).backward()
            opt.step()
    with torch.no_grad():
        ce = float(F.cross_entropy(probe(z_eval), y_eval))
        py = torch.bincount(y_fit, minlength=n_classes).float()
        py = py / py.sum()
        h_y = float(-(py * (py + 1e-12).log()).sum())
    return max(0.0, h_y - ce)


def partition_ceiling(i_code, n_classes, h_y=None):
    """[LOOSE 2-3x] Accuracy ceiling of a decoder that sees only a code carrying ``i_code`` nats."""
    return 1.0 - eps_for_rib(i_code, n_classes, h_y)


def selftest():
    """Pure-function checks against the campaign's recorded values. No data, no training."""
    import math as _m
    fails = []
    checks = [
        ("MNIST floor", abs(task_info_floor(0.918, 10, 2.3005) - 1.837) < 0.01),
        ("CIFAR floor", abs(task_info_floor(0.753, 10, 2.3024) - 1.201) < 0.01),
        ("MNIST interval top", abs(tightness_interval(1.97, task_info_floor(0.918, 10, 2.3005), 2.3005)[1]
                                   - 1.252) < 0.01),
        ("CIFAR interval top", abs(tightness_interval(1.60, task_info_floor(0.753, 10, 2.3024), 2.3024)[1]
                                   - 1.917) < 0.01),
        ("veto fires below the floor", veto(k_codes=4, rib=1.837)[0] == "IMPOSSIBLE"),
        ("veto passes above the floor", veto(k_codes=64, rib=1.837)[0] == "FEASIBLE"),
        ("balanced labels do not warn", balance_warning(_m.log(10), 10) is None),
        ("unbalanced labels warn", balance_warning(2.0, 10) is not None),
        ("floor is monotone in accuracy", task_info_floor(0.9, 10) > task_info_floor(0.5, 10)),
        ("floors are consistent", capacity_floors(1.837)["min_codebook_size"] >= 2),
    ]
    for name, ok in checks:
        if not ok:
            fails.append(name)
    return fails


# ---------------------------------------------------------------------------
# Bit-width allocation (zero training, like every measured function here).
#
# Recorded (pilot section 20, mixed precision vs one uniform width at the same average bits): at a
# 2.5-bit average, mixing widths reached the accuracy that a single uniform width needed 3.0 bits for,
# i.e. half a bit saved at equal accuracy, and the difference vanished at 3.0 bits and above. The win
# therefore lives at TIGHT average widths. QAT, which recovers bit starvation, is a TRAINING action and
# deliberately does not live in this module: the advisor measures and plans, it never trains.
# ---------------------------------------------------------------------------
BIT_GRID = [2, 3, 4, 5, 6, 8]


def quantise_weight(W, bits):
    """Symmetric per-tensor quantisation to `bits` bits (>=16 returns the tensor untouched)."""
    if bits >= 16:
        return W
    lv = 2 ** (bits - 1) - 1
    scale = W.abs().max() / max(lv, 1)
    return torch.clamp(torch.round(W / scale), -lv - 1, lv) * scale


def _weight_modules(model):
    return [m for m in model.modules()
            if getattr(m, "weight", None) is not None and m.weight.dim() >= 2]


def total_bits(model, bits):
    """Total stored bits of the quantised weights under a per-module width list."""
    return sum(m.weight.numel() * b for m, b in zip(_weight_modules(model), bits))


def average_bits(model, bits):
    """Average width over all quantised weights."""
    return total_bits(model, bits) / sum(m.weight.numel() for m in _weight_modules(model))


def accuracy_at_bits(model, x, y, bits):
    """[MEAS] Accuracy with the weights quantised to the given widths (forward pass only)."""
    mods = _weight_modules(model)
    saved = [m.weight.data.clone() for m in mods]
    try:
        with torch.no_grad():
            for m, b in zip(mods, bits):
                m.weight.data = quantise_weight(m.weight.data, b)
            return float((model(x).argmax(1) == y).float().mean())
    finally:
        for m, w in zip(mods, saved):
            m.weight.data = w


def plan_bit_allocation(model, x, y, target_avg, grid=None):
    """[MEAS] Zero-training bit-width plan at an average-width budget.

    Returns the uniform reference, the per-layer greedy proxy, and the whole-network search result,
    together with each arm's real average width and measured accuracy. A non-integer target cannot be
    matched by a single uniform width; compare the arms at the widths actually spent.
    """
    if torch is None:
        raise RuntimeError("plan_bit_allocation needs torch")
    grid = list(grid or BIT_GRID)
    mods = _weight_modules(model)
    n = len(mods)
    budget = int(sum(m.weight.numel() for m in mods) * target_avg)

    def greedy():
        curves = []
        for i in range(n):
            c = {}
            for b in grid:
                probe = [grid[-1]] * n
                probe[i] = b
                c[b] = accuracy_at_bits(model, x, y, probe)
            curves.append(c)
        bits = [grid[0]] * n
        while True:
            best = None
            for i in range(n):
                nxt = next((b for b in grid if b > bits[i]), None)
                if nxt is None:
                    continue
                trial = list(bits)
                trial[i] = nxt
                if total_bits(model, trial) > budget:
                    continue
                cost = total_bits(model, trial) - total_bits(model, bits)
                gain = (curves[i][nxt] - curves[i][bits[i]]) / max(cost, 1)
                if best is None or gain > best[0]:
                    best = (gain, i, nxt)
            if best is None:
                return bits
            bits[best[1]] = best[2]

    def search(start):
        bits = list(start)
        best = accuracy_at_bits(model, x, y, bits)
        for _ in range(5):
            improved = False
            for i in range(n):
                for b in grid:
                    if b == bits[i]:
                        continue
                    trial = list(bits)
                    trial[i] = b
                    if total_bits(model, trial) > budget:
                        continue
                    a = accuracy_at_bits(model, x, y, trial)
                    if a > best + 1e-5:
                        bits, best, improved = trial, a, True
            if not improved:
                break
        return bits

    ug = max((b for b in grid if total_bits(model, [b] * n) <= budget), default=grid[0])
    arms = {"uniform": [ug] * n, "greedy": greedy(), "mixed": search([ug] * n)}
    return {"arms": arms,
            "average": {k: average_bits(model, v) for k, v in arms.items()},
            "accuracy": {k: accuracy_at_bits(model, x, y, v) for k, v in arms.items()},
            "target_average": target_avg}


def main(argv=None):
    """CLI: the certified part of the advisor report. ``python -m capacity_lens.compress --acc 0.918``.

    Only the provable/archived sections are produced here: they need no model and no data. The measured
    sections (whole-network allocation, partition retention) need a trained model, so they live in the
    campaign-side driver rather than in the shipped package.
    """
    import argparse
    ap = argparse.ArgumentParser(prog="python -m capacity_lens.compress",
                                 description="compression budget advisor: certified guards and intervals")
    ap.add_argument("--acc", type=float, help="this model's test accuracy")
    ap.add_argument("--classes", type=int, default=10, help="number of classes")
    ap.add_argument("--h-y", dest="h_y", type=float, default=None,
                    help="empirical label entropy in nats (default ln C, valid only for balanced classes)")
    ap.add_argument("--i-lb", dest="i_lb", type=float, default=None,
                    help="task-relevant information of the representation, in nats")
    ap.add_argument("--k-codes", dest="k_codes", type=int, default=None, help="proposed bottleneck size")
    ap.add_argument("--capacity-nats", dest="capacity_nats", type=float, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        fails = selftest()
        print(f"[compress selftest] {len(fails)} failures")
        for f in fails:
            print("   " + f)
        return 1 if fails else 0
    if args.acc is None:
        ap.error("--acc is required (the report is about a model you already trained)")
    h_y = args.h_y if args.h_y is not None else math.log(args.classes)
    warn = balance_warning(h_y, args.classes)
    rib = task_info_floor(args.acc, args.classes, h_y)
    fl = capacity_floors(rib)
    print(f"[CERT] H(y)={h_y:.4f}  ln C={math.log(args.classes):.4f}  error={1 - args.acc:.4f}")
    print(f"[CERT] R_IB={rib:.4f} nats  ->  K_min={fl['min_codebook_size']}  "
          f"(ceil(e^R_IB)={math.exp(rib):.2f})")
    degenerate = []
    for d, q in fl["quant"].items():
        note = ""
        if q["min_levels"] < 1.05:
            note = "  <- degenerate (see the note below)"
            degenerate.append(d)
        print(f"[CERT] quant d={d}: min_bits/dim {q['min_bits_per_dim']:.3f} "
              f"({q['min_levels']:.1f} levels){note}")
    if degenerate:
        print(f"[CERT] note: the per-dimension bit floor degenerates at d={degenerate}. That is a property")
        print("       of the bound on low-information tasks (MNIST carries only ln 10 ~ 2.3 nats), not a")
        print("       defect: read K_min and the codebook floors instead of the per-dimension bit floor.")
    verdict, why = veto(args.k_codes, args.capacity_nats, rib)
    print(f"[CERT] veto: {verdict} -- {why}")
    if args.i_lb is not None:
        lo, hi = tightness_interval(args.i_lb, rib, h_y)
        print(f"[CERT] certified looseness in [{lo:.3f}, {hi:.3f}] -> the certificate is provably "
              f"within {hi:.2f}x of the best possible")
    if warn:
        print("[CERT] " + warn)
    print(f"[ARCH] the certified floor sits {FLOOR_VS_KNEE[0]:.0f}-{FLOOR_VS_KNEE[1]:.0f}x below the practical "
          f"knee; partition retention is {RETENTION_RANGE[0]:.2f}-{RETENTION_RANGE[1]:.2f} at K=64 and its "
          f"ceiling is {CEILING_LOOSENESS[0]:.0f}-{CEILING_LOOSENESS[1]:.0f}x loose; quote the protocol.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
