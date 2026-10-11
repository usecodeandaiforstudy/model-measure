# -*- coding: utf-8 -*-
"""CP3 supplementary test 2: functionally sensitive dynamics readouts (replacing L_info, which was contaminated by "equivalent-solution drift").

Background (the diagnostic conclusion of CP3 supplementary test 1, see falsified.md):
  L_info uses ds = RMS_b(g_b·Δθ) (**per-batch** root-mean-square of the gradient projection). For CNN-euclid s2,
  acc / H(Z) are exactly the same as for s1 (1.000 / 5.22 vs 5.24), yet L is 3.86 times theirs
  (279.7 vs 72.5), and the median dL is 4.77 (s0/s1 are 0.011/0.035).
  Root cause: after Adam converges it keeps displacing, while the single-batch gradient g_b is large -> noise×displacement dominates the arc length.
  -> L_info measures the bare geometric length in parameter space and does **not distinguish "movement that changes function" from "drift"**.

This experiment instead measures two functionally sensitive readouts (same training loop, bit-for-bit reproducing the CP3 teacher):
  (1) L_sys : Σ_t |ḡ_t · Δθ_t|, replacing the per-batch RMS with the **gradient window mean** ḡ,
      removing batch noise and keeping only the systematic first-order loss change.
  (2) L_func: Σ_epoch frac(pred_t != pred_{t-1}), the **total prediction flips** on a fixed probe set
      (the functional arc length, L0 version). Equivalent-solution drift does not change predictions -> contributes 0.

Criteria (prior to data):
  Key comparison CNN-euclid (rho=3.00) vs CNN-cosine (rho=2.00).
  Requires both (a) within-cell stability (max range < between-cell difference) and (b) between-cell separability (difference > 0).
  If the s2 outlier is eliminated and the two cells are separable -> functional sensitization succeeds; otherwise this direction also fails.

Execution: python experiments/cp3_dyn.py --all ; --cell CNN-euclid ; --report
"""
import argparse
import json
import os
import sys
import time
import traceback

import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
_DATA = os.path.join(_HERE, "data")
sys.path.append(_PKG)
sys.path.append(_HERE)

from capacity_lens import train_one_epoch, evaluate, entropy_of   # noqa: E402
from cp3_demo import (build_model, make_data, CELLS, TEACHER_CODES,   # noqa: E402
                      cell_budget)

BASE = os.path.join(_DATA, "cp3", "8dfe5bd4")
JSONL = os.path.join(_DATA, "cp3_dyn.jsonl")
LOG = os.path.join(_DATA, "cp3_dyn.log")
N_PER_CLASS = 150          # full tier
MAX_BATCHES = 32           # gradient window B
PROBE_N = 512              # functional probe set size (one forward pass per epoch, cost negligible)

CP3_RHO = {
    "MLP-euclid": [1.85, 1.75, 1.68],
    "MLP-cosine": [2.04, None, 3.69],
    "CNN-euclid": [3.00, 3.00, 3.00],
    "CNN-cosine": [2.00, 2.00, 2.00],
}


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        self.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


try:
    _LOG = open(LOG, "w", encoding="utf-8")
except OSError:
    # An import must not require a writable data directory (offline replay, read-only checkouts). Fall back to
    # devnull, NOT to None: these scripts tee stdout/stderr through _LOG, and a None sink makes every print
    # fail at interpreter shutdown with exit code 120 (found by the selftest loop that CI runs).
    _LOG = open(os.devnull, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _LOG)
sys.stderr = _Tee(sys.__stderr__, _LOG)


def _redirect_log(path):
    """Switch the log side-write target to a separate file (to avoid overwriting each other with a process still training in the background)."""
    global _LOG
    try:
        _LOG.close()
    except Exception:
        pass
    _LOG = open(path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _LOG)
    sys.stderr = _Tee(sys.__stderr__, _LOG)


class SysGradArc:
    """L_sys = Σ |mean_b(g_b) · Δθ|: removes batch noise using the gradient window mean.

    The only difference from FisherArcLength: the per-batch RMS of the projection sqrt(mean(g_b·Δθ)²)
    is replaced by the **absolute value of the mean projection** |mean(g_b)·Δθ|. The former contains batch noise; the latter keeps only the systematic term.
    """

    def __init__(self, model, max_batches=MAX_BATCHES):
        self.model = model
        self.max_batches = max_batches
        self._grads = []
        self._theta = None
        self._params = None
        self.total = 0.0

    def attach(self, optimizer):
        def pre_hook(opt, args, kwargs):
            with torch.no_grad():
                ps = [p for p in self.model.parameters() if p.grad is not None]
                if not ps:
                    return
                self._params = ps
                self._grads.append(torch.cat(
                    [p.grad.detach().reshape(-1) for p in ps]).float().cpu())
                if len(self._grads) > self.max_batches:
                    self._grads.pop(0)
                self._theta = torch.cat(
                    [p.detach().reshape(-1) for p in ps]).float().cpu()

        def post_hook(opt, args, kwargs):
            if self._theta is None or not self._params:
                return
            theta_new = torch.cat(
                [p.detach().reshape(-1) for p in self._params]).float().cpu()
            dtheta = theta_new - self._theta
            grads = [g for g in self._grads if g.numel() == dtheta.numel()]
            if grads:
                g_bar = torch.stack(grads).mean(dim=0)
                self.total += abs(float(g_bar @ dtheta))
            self._theta = None

        optimizer.register_step_pre_hook(pre_hook)
        optimizer.register_step_post_hook(post_hook)
        return self


@torch.no_grad()
def probe_pred(model, x):
    model.eval()
    out, _ = model(x)
    model.train()
    return out.argmax(dim=1)


def run_one(cell, c_idx, s):
    name, arch, metric = cell["name"], cell["arch"], cell["metric"]
    budget = cell_budget(arch, quick=False)
    epochs, lr = budget["epochs"], budget["lr"]

    torch.manual_seed(14000 + 100 * c_idx + s)      # consistent with cp3_demo.train_teacher
    (xtr, ytr), (xte, yte) = make_data(arch, N_PER_CLASS, s)
    train_loader = DataLoader(list(zip(xtr, ytr)), batch_size=128, shuffle=True)
    test_loader = DataLoader(list(zip(xte, yte)), batch_size=256)
    model = build_model(arch, TEACHER_CODES, metric)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    arc = SysGradArc(model).attach(opt)

    probe_x = xte[:PROBE_N]
    pred_prev = probe_pred(model, probe_x)
    l_func = 0.0
    sys_hist, func_hist = [], []
    prev_sys = 0.0

    t0 = time.time()
    for e in range(epochs):
        train_one_epoch(model, train_loader, opt)
        pred = probe_pred(model, probe_x)
        flip = float((pred != pred_prev).float().mean())
        l_func += flip
        pred_prev = pred
        sys_hist.append(arc.total - prev_sys)
        prev_sys = arc.total
        func_hist.append(flip)
        if (e + 1) % max(1, epochs // 4) == 0 or e == 0:
            print(f"    ep{e + 1}/{epochs}  L_sys={arc.total:8.3f}  "
                  f"L_func={l_func:.3f}  flip={flip:.4f}  {time.time() - t0:.0f}s",
                  flush=True)
    dur = time.time() - t0

    acc, idx = evaluate(model, test_loader)
    h_z = entropy_of(idx[0], TEACHER_CODES)
    ck = torch.load(os.path.join(BASE, f"{name}_teacher_s{s}.pt"),
                    map_location="cpu", weights_only=True)
    return dict(cell=name, s=s, arch=arch, metric=metric, epochs=epochs,
                l_sys=arc.total, l_func=l_func,
                d_sys_last=sys_hist[-1], d_func_last=func_hist[-1],
                sys_hist=sys_hist, func_hist=func_hist,
                acc=acc, h_z=h_z, acc_ckpt=ck["acc"], hz_ckpt=ck["h_z"],
                sec=dur, rho=CP3_RHO[name][s])


def run_and_log(cell, c_idx, s):
    r = run_one(cell, c_idx, s)
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
    print(f"--- {r['cell']} s{s} ---", flush=True)
    print(f"  acc={r['acc']:.4f} (archived {r['acc_ckpt']:.4f})  "
          f"H(Z)={r['h_z']:.3f} (archived {r['hz_ckpt']:.3f})  "
          f"L_sys={r['l_sys']:.2f}  L_func={r['l_func']:.4f}  "
          f"rho={r['rho']}  {r['sec']:.0f}s", flush=True)


FULL_EPOCHS = {"mlp": 120, "cnn": 40}


def load_rows():
    if not os.path.exists(JSONL):
        return []
    last = {}
    with open(JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["epochs"] != FULL_EPOCHS[r["arch"]]:
                continue
            last[(r["cell"], r["s"])] = r
    return [last[k] for k in sorted(last)]


def report():
    rows = load_rows()
    print("=" * 92)
    print("[CP3 supplementary test 2] functionally sensitive dynamics readouts vs rho")
    print("  L_sys = Σ|mean_b(g_b)·Δθ| (batch noise removed)")
    print("  L_func = Σ_epoch prediction flip rate (equivalent-solution drift contributes 0)")
    print("=" * 92)
    print(f"{'grid':<12}{'s':>3}{'L_sys':>11}{'L_func':>10}{'last dL_sys':>12}"
          f"{'last flip':>10}{'acc':>8}{'rho':>7}")
    for r in rows:
        rho = f"{r['rho']:.2f}" if r["rho"] else "  -"
        print(f"{r['cell']:<12}{r['s']:>3}{r['l_sys']:>11.3f}{r['l_func']:>10.4f}"
              f"{r['d_sys_last']:>12.4f}{r['d_func_last']:>10.4f}"
              f"{r['acc']:>8.4f}{rho:>7}")

    print("\n" + "=" * 92)
    print("Between-grid summary (mean ± half range)")
    print("=" * 92)
    for cell in CELLS:
        rs = [r for r in rows if r["cell"] == cell["name"]]
        if not rs:
            print(f"  {cell['name']:<12} [missing]")
            continue

        def ms(k):
            v = [r[k] for r in rs]
            return f"{sum(v) / len(v):.4f}±{(max(v) - min(v)) / 2:.4f}"

        print(f"  {cell['name']:<12} L_sys={ms('l_sys'):>20}  "
              f"L_func={ms('l_func'):>18}")

    print("\n" + "=" * 92)
    print("Key comparison: CNN-euclid vs CNN-cosine (rho 3.00 vs 2.00)")
    print("=" * 92)
    for k, label in (("l_sys", "L_sys"), ("l_func", "L_func"),
                     ("d_sys_last", "last dL_sys"), ("d_func_last", "last flip")):
        e = [r[k] for r in rows if r["cell"] == "CNN-euclid"]
        c = [r[k] for r in rows if r["cell"] == "CNN-cosine"]
        if not e or not c:
            print(f"  {label:<14} [missing data]")
            continue
        me, mc = sum(e) / len(e), sum(c) / len(c)
        diff = abs(me - mc)
        spread = max(max(e) - min(e), max(c) - min(c))
        rel = diff / max(abs(me), abs(mc), 1e-12)
        tag = "separable" if diff > spread and rel > 0.1 else "overlap (inseparable)"
        print(f"  {label:<14} euclid={me:9.4f}  cosine={mc:9.4f}  "
              f"gap={diff:8.4f} ({rel * 100:5.1f}%)  max spread={spread:8.4f}  -> {tag}")

    print("\n" + "=" * 92)
    print("Reproduction correctness")
    print("=" * 92)
    da = max(abs(r["acc"] - r["acc_ckpt"]) for r in rows)
    dh = max(abs(r["h_z"] - r["hz_ckpt"]) for r in rows)
    print(f"  acc max deviation {da:.4f}; H(Z) max deviation {dh:.4f}"
          f"  -> {'bit-for-bit reproduction' if da < 1e-6 and dh < 1e-6 else 'deviation found; check needed'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        _redirect_log(os.path.join(_DATA, "cp3_dyn_report.txt"))
        report()
        _LOG.flush()
        return
    if args.all:
        done = {(r["cell"], r["s"]) for r in load_rows()}
        for c_idx, c in enumerate(CELLS):
            for s in range(3):
                if (c["name"], s) in done:
                    print(f"[skip] {c['name']} s{s} (exists)", flush=True)
                    continue
                run_and_log(c, c_idx, s)
        return
    if args.cell and args.seed is not None:
        c_idx = [c["name"] for c in CELLS].index(args.cell)
        run_and_log(CELLS[c_idx], c_idx, args.seed)
    elif args.cell:
        c_idx = [c["name"] for c in CELLS].index(args.cell)
        for s in range(3):
            run_and_log(CELLS[c_idx], c_idx, s)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        _LOG.flush()
        raise
    _LOG.flush()
