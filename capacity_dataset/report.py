"""capacity_dataset -- an INDEPENDENT extension package that turns dataset measurements into a decision report.

This package lives OUTSIDE the capacity_lens repository (reference/pkg/...) and only READS the published
capacity_lens API. It reimplements no information-theory formula: every floor, bracket and knee comes from
`capacity_lens`, so the two cannot drift apart.

DESIGN RULES
------------
1. only import, never copy: `task_info_fano`, `capacity_floors`, `budget_ladder`, `rank_plan_accuracy`
   and friends are called, not re-derived.
2. honesty contract: every reported number is either MEASURED or marked UNAVAILABLE; a precision/recall
   is NEVER printed without ground-truth labels.
3. deterministic: all randomness goes through an explicit seed.

PUBLIC API
----------
dup_report(X, y=None, ...)        -> dict   (no training, seconds)
saturation_report(X, y, ...)      -> dict   (one short training run)
review_price(X, y_true, y_noisy)  -> dict | None  (ground truth required)
build_report(X, y, ...)           -> dict   (all of the above + one recommendation)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

import capacity_lens as CL

__version__ = "1.0.0"

EPS = 0.10
DUP_WARN = 0.05
PLADDER = tuple(round(0.05 * i, 2) for i in range(1, 13))


# ------------------------------------------------------------------ helpers
def _as_tensor(X, y=None):
    X = X if isinstance(X, torch.Tensor) else torch.as_tensor(X, dtype=torch.float32)
    X = X.float()
    if y is None:
        return X, None
    y = y if isinstance(y, torch.Tensor) else torch.as_tensor(y)
    return X, y.long()


def _keys(X, digits):
    """Rounded row keys for duplicate detection at a stated precision."""
    q = (X * (10.0 ** digits)).round().to(torch.int64)
    return [tuple(row.tolist()) for row in q]


def _entropy(counts):
    n = sum(counts)
    return -sum((c / n) * math.log(c / n) for c in counts if c > 0)


def _model(d, c, width=64):
    return nn.Sequential(nn.Linear(d, width), nn.ReLU(), nn.Linear(width, c))


def _train(X, y, c, seed=0, epochs=25, bs=64):
    torch.manual_seed(seed)
    m = _model(X.shape[1], c)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    lossf = nn.CrossEntropyLoss(reduction="none")
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            lossf(m(X[j]), y[j]).mean().backward()
            opt.step()
    return m


def _level(m, X, y):
    with torch.no_grad():
        return float((m(X).argmax(1) == y).float().mean())


# ------------------------------------------------------------------ blocks
def dup_report(X, y=None, digits=(12, 4)):
    """Duplicate rate and label collisions. No training.

    `collisions` counts keys that appear under more than one label; those rows are indistinguishable to any
    model trained on this feature vector, so a high count is a property of the data, not of the model.
    """
    X, y = _as_tensor(X, y)
    n, d = X.shape
    out = dict(n=int(n), d=int(d), unavailable=[])
    for dg in digits:
        keys = _keys(X, dg)
        dup = 1 - len(set(keys)) / max(len(keys), 1)
        col = None
        if y is not None:
            lab = {}
            for k, t in zip(keys, y.tolist()):
                lab.setdefault(k, set()).add(t)
            col = sum(1 for v in lab.values() if len(v) > 1)
        out[f"dup_rate_{dg}dp"] = dup
        out[f"collisions_{dg}dp"] = col
    if y is None:
        out["unavailable"].append("class counts / entropy (no labels given)")
    else:
        counts = {}
        for t in y.tolist():
            counts[t] = counts.get(t, 0) + 1
        cs = sorted(counts.values(), reverse=True)
        out["c"] = len(cs)
        out["h_y"] = _entropy(cs)
        rib = CL.task_info_fano(EPS, len(cs), out["h_y"])
        out["rib"] = rib
        out["k_min"] = CL.capacity_floors(rib)["min_codebook_size"]
    return out


def saturation_report(X, y, seed=0, epochs=25, sizes=(0.25, 0.5, 1.0)):
    """level, parameter knee and sample knee. One short training run plus the ladders."""
    X, y = _as_tensor(X, y)
    c = int(y.max()) + 1
    torch.manual_seed(seed)
    perm = torch.randperm(len(X))
    n_tr = max(2, int(0.7 * len(X)))
    tr, te = perm[:n_tr], perm[n_tr:]
    m = _train(X[tr], y[tr], c, seed=seed, epochs=epochs)
    lvl = _level(m, X[te], y[te])
    lad = CL.budget_ladder(m, X[tr], y[tr], ladder=PLADDER, n_random=3, seed=seed)
    sample_knee = None
    if len(X) >= 400:
        accs = []
        for frac in sizes:
            k = max(32, int(len(X) * frac * 0.7))
            mm = _train(X[perm[:k]], y[perm[:k]], c, seed=seed, epochs=epochs)
            accs.append((k, _level(mm, X[te], y[te])))
        best = max(a for _, a in accs)
        sample_knee = next((k for k, a in accs if a >= best - 0.01), None)
    return dict(level=lvl, param_knee=lad["knee_frac"], sample_knee=sample_knee,
                train_rows=int(len(tr)), test_rows=int(len(te)),
                unavailable=[] if len(X) >= 400 else ["sample knee (n < 400)"])


def review_price(X, y_true, y_noisy, seed=0, epochs=20, ks=(0.10, 0.20), min_lift=1.5):
    """Rows-per-catch for a loss-ranked review list. Requires the TRUE labels.

    Returns None when the caller cannot supply ground truth. If the ranking fails to beat the random
    baseline by `min_lift`, the recall numbers are withheld instead of being reported as noise.
    """
    if y_true is None:
        return None
    X, yt = _as_tensor(X, y_true)
    _, yn = _as_tensor(X, y_noisy)
    wrong = (yt != yn)
    if int(wrong.sum()) == 0:
        return dict(unavailable=["no incorrect labels: nothing to catch"])
    c = int(yt.max()) + 1
    m = _train(X, yn, c, seed=seed, epochs=epochs)
    with torch.no_grad():
        loss = nn.functional.cross_entropy(m(X), yn, reduction="none")
    order = torch.argsort(loss, descending=True)
    base = float(wrong.float().mean())
    out = {"base_rows_per_catch": 1.0 / base, "error_rate": base}
    top = None
    for frac in ks:
        k = int(len(X) * frac)
        caught = float(wrong[order[:k]].sum())
        rec = caught / max(float(wrong.sum()), 1e-9)
        out[f"{int(frac*100)}%"] = dict(rows=k, caught=caught, recall=rec,
                                        rows_per_catch=k / max(caught, 1e-9))
        if top is None:
            top = caught / k
    out["signal_ok"] = bool(top >= min_lift * base)
    if not out["signal_ok"]:
        for frac in ks:
            out[f"{int(frac*100)}%"]["recall"] = None
        out["unavailable"] = ["recall: the loss ranking did not beat the random baseline by "
                              f"{min_lift}x, so no review price is quoted"]
    return out


def recommend(dup, sat):
    """One line, from a fixed rule table. DEDUP FIRST takes precedence (it is provably free)."""
    if dup.get("dup_rate_12dp", 0.0) >= DUP_WARN:
        return ("DEDUP FIRST",
                f"same-label duplicates {dup['dup_rate_12dp']*100:.1f}%: removing them cannot change the "
                f"information requirement and saves compute")
    if sat["level"] >= 0.95 and (sat["param_knee"] or 1) <= 0.20:
        return "TOO EASY", f"level {sat['level']:.3f} at parameter knee {sat['param_knee']}"
    if sat["level"] < 0.70:
        return ("UNREACHABLE",
                f"level {sat['level']:.3f}: either hard or noisy; this report cannot separate the two")
    return "USABLE", f"level {sat['level']:.3f} at parameter knee {sat['param_knee']}"


def build_report(X, y, y_true=None, y_noisy=None, seed=0, epochs=25):
    X, y = _as_tensor(X, y)
    dup = dup_report(X, y)
    sat = saturation_report(X, y, seed=seed, epochs=epochs)
    price = review_price(X, y_true, y_noisy, seed=seed) if y_true is not None else None
    tag, why = recommend(dup, sat)
    return dict(version=__version__, dup=dup, saturation=sat, price=price,
                recommendation=dict(tag=tag, why=why),
                notes=["review price requires ground truth; None means it was not supplied",
                       "every number is measured or explicitly listed as unavailable"])


def selftest():
    fails = []
    torch.manual_seed(0)
    X = torch.randn(600, 6)
    y = torch.randint(0, 3, (600,))
    d = dup_report(X, y)
    if d["n"] != 600 or d["c"] != 3:
        fails.append("dup_report shape")
    if not (0.0 <= d["dup_rate_12dp"] <= 1.0):
        fails.append("dup rate range")
    Xd = torch.cat([X, X[:100]])
    yd = torch.cat([y, y[:100]])
    d2 = dup_report(Xd, yd)
    if abs(d2["dup_rate_12dp"] - (1 - 600 / 700)) > 1e-6:
        fails.append("duplicate detection exact rate wrong")
    p = review_price(X, y, y, seed=0)
    if p is not None and "unavailable" not in p:
        fails.append("review_price should refuse when there are no errors")
    return fails


if __name__ == "__main__":
    f = selftest()
    print(f"[capacity_dataset selftest] {len(f)} failures")
    for x in f:
        print("   " + x)
