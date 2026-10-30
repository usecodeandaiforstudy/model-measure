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


# ---------------------------------------------------------------------------
# Rank allocation (zero training) -- the capability validated locally on 20+ instances:
#   free screen: a layer is NEVER pinned below MIN_RANK, because rank starvation is irreversible
#                (measured -0.205 retention, 0 of 6 seeds recovered by fine-tuning);
#   ranking    : whole-network accuracy at the truncated weights ranks candidates so that the top-3
#                contain the exhaustive best (20/20 eligible instances, gap +0.0000 every time,
#                Spearman rho median 0.90), which is 5.3x fewer fine-tunes than a full grid (16 -> 3).
# The budget is counted over the Linear layers only, and the water-filling reference spends it in full:
# a single common rank is dragged down by the smallest cap and silently under-spends (measured defect).
# ---------------------------------------------------------------------------
MIN_RANK = 2


def rank_parameter_count(shapes, ranks):
    """Parameters a low-rank allocation stores (``r * (in + out) + out`` per layer)."""
    return sum(r * (a + b) + b for r, (a, b) in zip(ranks, shapes))


def _linear_layers(model):
    # `nn` is only imported inside functions in this module, so reference the class through `torch`.
    return [m for m in model.modules() if isinstance(m, torch.nn.Linear)]


def uniform_rank_allocation(shapes, budget, caps=None, min_rank=MIN_RANK):
    """Budget-matched reference allocation: water-fill rank into every layer that still has headroom."""
    caps = [min(s) for s in shapes] if caps is None else list(caps)
    ranks = [min(min_rank, c) for c in caps]
    while True:
        trial = [min(c, r + 1) if r < c else r for r, c in zip(ranks, caps)]
        if trial == ranks or rank_parameter_count(shapes, trial) > budget:
            return ranks
        ranks = trial


def rank_plan_accuracy(model, x, y, ranks, batch_size=256):
    """[MEAS] Accuracy with every Linear replaced by its rank-truncated factors; the model is restored.

    Weights are restored in a ``finally`` block and the inputs are moved to the model's device, so this
    neither leaves the caller's model mutated nor fails on a GPU model handed CPU tensors.
    """
    layers = _linear_layers(model)
    if not layers:
        raise ValueError("rank_plan_accuracy needs at least one nn.Linear layer")
    saved = [l.weight.data.clone() for l in layers]
    device = layers[0].weight.device
    if x.device != device:
        x, y = x.to(device), y.to(device)
    try:
        with torch.no_grad():
            for l, r in zip(layers, ranks):
                U, S, V = torch.svd_lowrank(l.weight.data, q=min(int(r), min(l.weight.shape)), niter=2)
                l.weight.data = (U * S) @ V.t()
            correct = 0
            for i in range(0, x.shape[0], batch_size):
                correct += int((model(x[i:i + batch_size]).argmax(1) == y[i:i + batch_size]).sum())
        return correct / x.shape[0]
    finally:
        for l, w in zip(layers, saved):
            l.weight.data = w


def plan_rank_allocation(model, x, y, budget_params, min_rank=MIN_RANK, max_sweeps=5):
    """[MEAS] Zero-training rank allocation by coordinate descent on whole-network accuracy.

    Search only on the tensors you pass in (use a validation split, never the test set). Each step moves
    one layer's rank by +-1, keeps the allocation inside ``budget_params`` and never below ``min_rank``,
    and accepts a move only when the measured accuracy improves. Returns the plan, the budget-matched
    uniform reference and both scores, so the caller can see whether the search paid off.
    """
    layers = _linear_layers(model)
    if not layers:
        raise ValueError("plan_rank_allocation needs at least one nn.Linear layer")
    shapes = [(l.weight.shape[1], l.weight.shape[0]) for l in layers]
    caps = [min(s) for s in shapes]
    start = uniform_rank_allocation(shapes, budget_params, caps, min_rank)
    ranks = list(start)
    best = rank_plan_accuracy(model, x, y, ranks)
    uniform_score = best
    for _ in range(max_sweeps):
        improved = False
        for i in range(len(ranks)):
            for delta in (1, -1):
                cand = min(max(ranks[i] + delta, min(min_rank, caps[i])), caps[i])
                if cand == ranks[i]:
                    continue
                trial = list(ranks)
                trial[i] = cand
                if rank_parameter_count(shapes, trial) > budget_params:
                    continue
                score = rank_plan_accuracy(model, x, y, trial)
                if score > best + 1e-9:
                    ranks, best, improved = trial, score, True
        if not improved:
            break
    return dict(ranks=ranks, uniform=start, score=float(best), uniform_score=float(uniform_score),
                budget_params=int(budget_params), params=rank_parameter_count(shapes, ranks),
                shapes=shapes, caps=caps, min_rank=int(min_rank))


# ---------------------------------------------------------------------------
# STRATEGY LAYER -- heuristic on top of the certified measurements. NOT CERTIFIED.
#
#   Everything below reuses the strict primitives above and changes none of them. The strict side can
#   only say "impossible" or "undecided": in the wide band where a configuration is information-feasible
#   but practically useless, no bound decides. The honest cheap move there is to RANK candidates and let
#   the user fine-tune the few at the top.
#
#   Evidence (measured locally, small samples, one task family -- never quoted as a bound):
#     retention_curve   identical to the per-K measurement (max |delta retention| = 0.00000000 on real
#                       features) at ~2.2x the speed for a multi-K curve
#     shortlist         the top-3 by the zero-training score contained the exhaustive best on the one
#                       non-flat instance examined (gap +0.0000); flat instances are reported as flat
#     budget_ladder     the knee sat at 0.10x of full-rank parameters on both instances examined;
#                       lambda in [0.05, 0.10] bought 0.10x-0.15x for 0.5-0.75 accuracy points, while 0.05x
#                       collapsed by ~13 points
#   None of these numbers is a guarantee, and no output of this section may be called a bound.
# ---------------------------------------------------------------------------
STRATEGY_DISCLAIMER = (
    "STRATEGY / NOT CERTIFIED: this is a heuristic built on the strict measurements. It does not prove "
    "optimality and must not be quoted as a bound. The strict outputs are unchanged."
)
FLAT_TOL = 0.01        # if the best is within this of uniform, the allocation question has no content
REF_FLOOR = 0.4        # below this reference accuracy the score is noise (same gate as the campaign)

_PROBE_CACHE = {}


def _cache_key(*tensors):
    parts = []
    for t in tensors:
        flat = t.reshape(-1)
        sample = flat[:4].tolist() + flat[-4:].tolist() if flat.numel() >= 8 else flat.tolist()
        parts.append((tuple(t.shape), str(t.dtype), int(t.numel()),
                      tuple(round(float(v), 6) for v in sample)))
    return tuple(parts)


def retention_curve(z_fit, y_fit, z_eval, y_eval, ks, n_classes, seed=0):
    """[MEAS] Retention for several code counts with ONE probe training.

    The probe depends on (z, y) only, never on K, so a curve over several K does not need to retrain it.
    The numbers equal calling ``partition_retention`` once per K (measured: max |delta| = 0 on real
    features). Returns ``{K: {...}}`` plus ``"_probe_seconds"``.
    """
    if torch is None:
        raise RuntimeError("retention_curve needs torch")
    import time                          # local import: the module keeps its top-level imports minimal
    ks = [int(k) for k in ks]
    key = (_cache_key(z_fit, y_fit, z_eval, y_eval), int(n_classes), int(seed))
    t0 = time.time()
    if key in _PROBE_CACHE:
        i_rep = _PROBE_CACHE[key]
    else:
        i_rep = float(probe_lower_bound(z_fit, y_fit, z_eval, y_eval, n_classes, seed=seed))
        _PROBE_CACHE[key] = i_rep
    probe_seconds = time.time() - t0
    out = {}
    for k in ks:
        t1 = time.time()
        ctr = kmeans(z_fit, k, seed=seed)          # same seed as partition_retention
        idx_f = torch.cdist(z_fit, ctr).argmin(1)
        idx_e = torch.cdist(z_eval, ctr).argmin(1)
        i_code = float(mutual_info_of(idx_e, y_eval, k, n_classes))
        out[k] = dict(i_code=i_code, i_rep=i_rep,
                      retention=(i_code / i_rep if i_rep > 0 else float("nan")),
                      seconds=time.time() - t1)
    out["_probe_seconds"] = probe_seconds
    return out


def _same_budget_candidates(shapes, budget, n_random=14, seed=0):
    """Information-free candidate generator at a fixed budget: uniform, greedy proxy, seeded randoms."""
    import random                       # local import: this module keeps its top-level imports minimal
    caps = [min(s) for s in shapes]
    rng = random.Random(seed)
    out = {"uniform": uniform_rank_allocation(shapes, budget, caps, MIN_RANK)}
    proxy = [min(MIN_RANK, c) for c in caps]
    order = sorted(range(len(caps)), key=lambda i: -caps[i])
    while True:
        placed = False
        for i in order:
            if proxy[i] >= caps[i]:
                continue
            trial = list(proxy)
            trial[i] += 1
            if rank_parameter_count(shapes, trial) <= budget:
                proxy[i] += 1
                placed = True
        if not placed:
            break
    out["proxy"] = proxy
    for j in range(n_random):
        trial = [rng.randint(MIN_RANK, max(MIN_RANK, c)) for c in caps]
        while rank_parameter_count(shapes, trial) > budget:
            # Stop when nothing can be shrunk: a budget below the minimum allocation (for example a
            # model with one small Linear layer) used to loop forever here instead of yielding no
            # candidate at all. The caller then reports allocation_freedom = False.
            if all(t <= MIN_RANK for t in trial):
                break
            i = rng.randrange(len(caps))
            if trial[i] > MIN_RANK:
                trial[i] -= 1
        if rank_parameter_count(shapes, trial) <= budget:
            out["rand%d" % j] = trial
    return out


def shortlist_rank_allocations(model, x, y, budget_params, k=3, n_random=14, seed=0,
                               include_descent=True):
    """[STRATEGY] Rank same-budget allocations and return the top ``k``. Zero training, not a bound.

    Score = ``rank_plan_accuracy`` at the truncated weights (one forward pass per candidate); the caller
    fine-tunes only the shortlist, which is where the saving comes from. Flags: ``allocation_freedom``
    (False when every candidate collapses to the same ranks, so the question is empty), ``flat`` (best
    within ``FLAT_TOL`` of uniform) and ``unreliable_note`` (the reference-accuracy gate).
    """
    layers = _linear_layers(model)
    if not layers:
        raise ValueError("shortlist_rank_allocations needs at least one nn.Linear layer")
    shapes = [(l.weight.shape[1], l.weight.shape[0]) for l in layers]
    cands = _same_budget_candidates(shapes, budget_params, n_random=n_random, seed=seed)
    if include_descent:
        try:
            cands["descent"] = plan_rank_allocation(model, x, y, budget_params, min_rank=MIN_RANK)["ranks"]
        except Exception:                                  # noqa: BLE001
            pass
    scored = []
    for name, ranks in cands.items():
        if min(ranks) < MIN_RANK or rank_parameter_count(shapes, ranks) > budget_params:
            continue
        scored.append(dict(name=name, ranks=list(ranks),
                           score=float(rank_plan_accuracy(model, x, y, ranks)),
                           params=int(rank_parameter_count(shapes, ranks))))
    scored.sort(key=lambda r: -r["score"])
    uniform = next((r for r in scored if r["name"] == "uniform"), None)
    best = scored[0] if scored else None
    distinct = len({tuple(r["ranks"]) for r in scored})
    return dict(disclaimer=STRATEGY_DISCLAIMER, shortlist=scored[:k], best=best, uniform=uniform,
                n_candidates=len(scored), n_distinct=distinct,
                allocation_freedom=bool(distinct > 1),
                flat=bool(uniform and best and (best["score"] - uniform["score"]) < FLAT_TOL),
                unreliable_note=("reference accuracy below %.1f makes this score noise" % REF_FLOOR),
                budget_params=int(budget_params), min_rank=int(MIN_RANK))


def budget_ladder(model, x, y, ladder=(0.05, 0.10, 0.15, 0.25, 0.40, 0.60), n_random=10, seed=0):
    """[STRATEGY] Best allocation at each budget level, plus the knee. Zero training, not a bound.

    Inside one budget every candidate spends the same parameters; across budgets the curve has a slope,
    which is where cost and accuracy actually trade. ``knee_frac`` is the smallest budget whose score is
    within ``FLAT_TOL`` of the best score anywhere on the ladder.
    """
    layers = _linear_layers(model)
    if not layers:
        raise ValueError("budget_ladder needs at least one nn.Linear layer")
    shapes = [(l.weight.shape[1], l.weight.shape[0]) for l in layers]
    full = rank_parameter_count(shapes, [min(s) for s in shapes])
    curve = []
    for frac in ladder:
        budget = max(2 * MIN_RANK * len(shapes), int(full * float(frac)))
        best = None
        for name, ranks in _same_budget_candidates(shapes, budget, n_random=n_random, seed=seed).items():
            if min(ranks) < MIN_RANK or rank_parameter_count(shapes, ranks) > budget:
                continue
            s = rank_plan_accuracy(model, x, y, ranks)
            if best is None or s > best["score"]:
                best = dict(name=name, ranks=list(ranks), score=float(s),
                            params=int(rank_parameter_count(shapes, ranks)))
        if best is not None:
            curve.append(dict(frac=float(frac), **best))
    best_score = max((c["score"] for c in curve), default=float("nan"))
    knee = next((c for c in curve if c["score"] >= best_score - FLAT_TOL), None)
    return dict(disclaimer=STRATEGY_DISCLAIMER, curve=curve, full_params=int(full),
                best_score=best_score, knee_frac=(knee["frac"] if knee else None),
                knee_params=(knee["params"] if knee else None))


def choose_operating_point(curve, lam):
    """[STRATEGY] Budget level maximising ``accuracy - lam * (params / full_params)``.

    ``curve`` is ``budget_ladder(...)["curve"]``. ``lam = 0`` keeps the most accurate level, larger values
    prefer cheaper levels. It is a one-dimensional preference knob, not a decision the library can make.
    """
    if not curve:
        return None
    full = max(c["params"] for c in curve) or 1
    return max(curve, key=lambda c: c["score"] - float(lam) * (c["params"] / full))


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
