# -*- coding: utf-8 -*-
"""L_func sensitivity test: does the readout depend on the choice of probe set?

Motivation (the results of cp3_dyn):
  L_func (the accumulated prediction flips) is "separable" in the main CNN comparison (4.38 vs 1.37, difference 68.6%),
  but the margin is only 1.28x (difference 3.01 / max range 2.35), n=3. The possibility of a "coincidental probe set" must be ruled out.

Design: within a single training run, accumulate the L_func of 4 probe sets simultaneously (same model, same trajectory):
  P1_te512  = test set[0:512]     (the original setting, baseline)
  P2_te512b = test set[512:1024]  (same size, different samples -> tests "content sensitivity")
  P3_te1024 = test set[0:1024]    (larger sample -> tests "size sensitivity")
  P4_tr512  = train set[0:512]    (samples the model has seen -> tests "generalization vs memorization")

Criteria (prior to data):
  For each probe set, separately test the separability of the key comparison CNN-euclid vs CNN-cosine
  (difference > max range) and the same-direction relationship between L_func and rho across the four cells.
  If all probe sets agree -> the readout is robust to probe-set choice; if they disagree -> unstable, unusable.

Execution: python experiments/cp3_dyn_sens.py --all ; --report

Early-window test (--early / --early-report):
  Motivation: if the L_func accumulated over the first K epochs already separates the two CNN cells, then the reward cost drops
  from "full training" to "first K epochs", and it supports mid-training decisions (RL's unique advantage over BO).
  Criteria (prior to data, P1 probe set):
    zero overlap (min_euclid > max_cosine) -> early-window reward is feasible (single-seed discrimination);
    mean separable but per-seed overlapping -> multi-seed averaging needed (evaluation cost x n);
    means overlapping -> early window infeasible, reward needs full training.
  Data is written to cp3_dyn_early.jsonl (per-epoch curves), separate from the main sensitivity data;
  the final values are cross-checked against the main record (same-seed deterministic rerun, should be bit-for-bit identical).
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
JSONL = os.path.join(_DATA, "cp3_dyn_sens.jsonl")
EARLY_JSONL = os.path.join(_DATA, "cp3_dyn_early.jsonl")
LOG = os.path.join(_DATA, "cp3_dyn_sens.log")
N_PER_CLASS = 150
MAX_BATCHES = 32

PROBES = {
    "P1_te512": ("te", 0, 512),
    "P2_te512b": ("te", 512, 1024),
    "P3_te1024": ("te", 0, 1024),
    "P4_tr512": ("tr", 0, 512),
}
PROBE_KEYS = list(PROBES)

CP3_RHO = {
    "MLP-euclid": [1.85, 1.75, 1.68],
    "MLP-cosine": [2.04, None, 3.69],
    "CNN-euclid": [3.00, 3.00, 3.00],
    "CNN-cosine": [2.00, 2.00, 2.00],
}


def rho_of(cell_name, s):
    """Fetch the CP3-archived ground-truth rho; returns None when s exceeds the 3 measured seeds.

    Note: CP3 measured rho only for s0-s2. For s>=3 there is no ground truth, and they are used only to
    make the range estimate of L_func more solid (statistical dimension), not for the rho comparison.
    """
    lst = CP3_RHO.get(cell_name, [])
    return lst[s] if s < len(lst) else None


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
    global _LOG
    try:
        _LOG.close()
    except Exception:
        pass
    _LOG = open(path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _LOG)
    sys.stderr = _Tee(sys.__stderr__, _LOG)


class SysGradArc:
    """L_sys (auxiliary readout, defined as in cp3_dyn)."""

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


def build_probes(xtr, xte):
    out = {}
    for k, (src, a, b) in PROBES.items():
        arr = xtr if src == "tr" else xte
        out[k] = arr[a:b]
    return out


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

    probes = build_probes(xtr, xte)
    preds = {k: probe_pred(model, v) for k, v in probes.items()}
    l_funcs = {k: 0.0 for k in probes}
    curves = {k: [] for k in probes}     # accumulate L_func per epoch (for the early-window analysis)

    t0 = time.time()
    for e in range(epochs):
        train_one_epoch(model, train_loader, opt)
        for k, v in probes.items():
            p = probe_pred(model, v)
            l_funcs[k] += float((p != preds[k]).float().mean())
            preds[k] = p
            curves[k].append(l_funcs[k])
        if (e + 1) % max(1, epochs // 4) == 0 or e == 0:
            print(f"    ep{e + 1}/{epochs}  " + "  ".join(
                f"{k}={l_funcs[k]:.2f}" for k in PROBE_KEYS)
                + f"  L_sys={arc.total:.1f}  {time.time() - t0:.0f}s", flush=True)
    dur = time.time() - t0

    acc, idx = evaluate(model, test_loader)
    h_z = entropy_of(idx[0], TEACHER_CODES)
    ck_path = os.path.join(BASE, f"{name}_teacher_s{s}.pt")
    # for s>=3 there is no CP3 archived checkpoint, so skip the reproduction check (acc_ckpt is None, excluded in report)
    ck = (torch.load(ck_path, map_location="cpu", weights_only=True)
          if os.path.exists(ck_path) else None)
    return dict(cell=name, s=s, arch=arch, metric=metric, epochs=epochs,
                l_sys=arc.total, l_funcs=l_funcs, curve=curves,
                acc=acc, h_z=h_z,
                acc_ckpt=ck["acc"] if ck else None,
                hz_ckpt=ck["h_z"] if ck else None,
                sec=dur, rho=rho_of(name, s))


def run_and_log(cell, c_idx, s, path=JSONL):
    r = run_one(cell, c_idx, s)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
    ckpt_txt = (f" (archived {r['acc_ckpt']:.4f})" if r["acc_ckpt"] is not None
                else " (no archive, skip verification)")
    print(f"--- {r['cell']} s{s} ---  " + "  ".join(
        f"{k}={r['l_funcs'][k]:.3f}" for k in PROBE_KEYS)
        + f"  acc={r['acc']:.4f}{ckpt_txt}  rho={r['rho']}",
        flush=True)


FULL_EPOCHS = {"mlp": 120, "cnn": 40}
WINDOWS = [1, 2, 5, 10, 20, 40]          # truncation points (first K epochs) for the early-window analysis


def load_rows(path=None):
    path = path or JSONL
    if not os.path.exists(path):
        return []
    last = {}
    with open(path, encoding="utf-8") as f:
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
    print("=" * 96)
    print("[L_func sensitivity test] 4 probe sets (accumulated within a single training run)")
    for k, (src, a, b) in PROBES.items():
        print(f"  {k:<11} = {'test set' if src == 'te' else 'train set'}[{a}:{b}]")
    print("=" * 96)
    print(f"{'grid':<12}{'s':>3}" + "".join(f"{k:>13}" for k in PROBE_KEYS)
          + f"{'acc':>9}{'rho':>7}")
    if not rows:
        print(f"{'grid':<12}{'s':>3}" + "".join(f"{k:>13}" for k in PROBE_KEYS)
              + f"{'acc':>9}{'rho':>7}")
        print("  [no data: training runs not yet complete]")
        print("=" * 96)
        return
    for r in rows:
        rho = f"{r['rho']:.2f}" if r["rho"] else "  -"
        print(f"{r['cell']:<12}{r['s']:>3}"
              + "".join(f"{r['l_funcs'][k]:>13.3f}" for k in PROBE_KEYS)
              + f"{r['acc']:>9.4f}{rho:>7}")

    for pk in PROBE_KEYS:
        print("\n" + "=" * 96)
        print(f"Probe set {pk}: between-grid summary and key comparison")
        print("=" * 96)
        for cell in CELLS:
            v = [r["l_funcs"][pk] for r in rows if r["cell"] == cell["name"]]
            if v:
                print(f"  {cell['name']:<12} {sum(v) / len(v):8.3f}"
                      f"±{(max(v) - min(v)) / 2:.3f}")
        e = [r["l_funcs"][pk] for r in rows if r["cell"] == "CNN-euclid"]
        c = [r["l_funcs"][pk] for r in rows if r["cell"] == "CNN-cosine"]
        if e and c:
            me, mc = sum(e) / len(e), sum(c) / len(c)
            diff = abs(me - mc)
            spread = max(max(e) - min(e), max(c) - min(c))
            rel = diff / max(abs(me), abs(mc), 1e-12)
            print(f"  -> CNN comparison: euclid={me:.3f} cosine={mc:.3f} "
                  f"gap={diff:.3f} ({rel * 100:.1f}%) max spread={spread:.3f} "
                  f"-> {'separable' if diff > spread else 'overlap'}")
        me_, mc_ = None, None
        for cell, target in (("MLP-euclid", "MLP-euclid"), ("MLP-cosine", "MLP-cosine")):
            v = [r["l_funcs"][pk] for r in rows if r["cell"] == cell]
            if v:
                if cell == "MLP-euclid":
                    me_ = (sum(v) / len(v), max(v) - min(v))
                else:
                    mc_ = (sum(v) / len(v), max(v) - min(v))
        if me_ and mc_:
            diff = abs(me_[0] - mc_[0])
            spread = max(me_[1], mc_[1])
            rel = diff / max(abs(me_[0]), abs(mc_[0]), 1e-12)
            print(f"  -> MLP comparison: euclid={me_[0]:.3f} cosine={mc_[0]:.3f} "
                  f"gap={diff:.3f} ({rel * 100:.1f}%) max spread={spread:.3f} "
                  f"-> {'separable' if diff > spread else 'overlap'}")

    print("\n" + "=" * 96)
    print("Reproduction correctness")
    print("=" * 96)
    ck_rows = [r for r in rows if r["acc_ckpt"] is not None]
    if not ck_rows:
        print("  [no archived checkpoint to compare]")
    else:
        da = max(abs(r["acc"] - r["acc_ckpt"]) for r in ck_rows)
        dh = max(abs(r["h_z"] - r["hz_ckpt"]) for r in ck_rows)
        extra = (f"  (compare s0-s{max(r['s'] for r in ck_rows)}, "
                 f"{len(rows) - len(ck_rows)} s>=3 records have no archive and are excluded)"
                 if len(ck_rows) < len(rows) else "")
        print(f"  acc max deviation {da:.4f}; H(Z) max deviation {dh:.4f}{extra}"
              f"  -> {'bit-for-bit reproduction' if da < 1e-6 and dh < 1e-6 else 'deviation found; check needed'}")


def early_report():
    rows = [r for r in load_rows(EARLY_JSONL) if r["arch"] == "cnn"]
    print("=" * 96)
    print("[early-window test] separability of the two CNN cells by L_func accumulated over the first K epochs (P1_te512)")
    print("  Criteria (prior to data):")
    print("    zero overlap min(euclid) > max(cosine)  -> early-window reward is feasible (single-seed discrimination)")
    print("    means separable but per-seed overlapping       -> multi-seed averaging needed (evaluation cost x n)")
    print("    means overlapping                            -> early window infeasible, reward needs full training")
    print("=" * 96)
    if not rows:
        print("  [no data: run the two CNN cells with --early first]")
        print("=" * 96)
        return
    for K in WINDOWS:
        e = [r["curve"]["P1_te512"][K - 1] for r in rows
             if r["cell"] == "CNN-euclid" and len(r["curve"]["P1_te512"]) >= K]
        c = [r["curve"]["P1_te512"][K - 1] for r in rows
             if r["cell"] == "CNN-cosine" and len(r["curve"]["P1_te512"]) >= K]
        if not e or not c:
            print(f"\nwindow K={K:<3} [missing data]")
            continue
        me, mc = sum(e) / len(e), sum(c) / len(c)
        diff = abs(me - mc)
        spread = max(max(e) - min(e), max(c) - min(c))
        rel = diff / max(abs(me), abs(mc), 1e-12)
        gap = min(e) - max(c)
        if gap > 0:
            tag = f"zero overlap (gap {gap:.3f}) -> early window feasible"
        elif diff > spread:
            tag = "means separable, per-seed overlapping -> multi-seed averaging needed"
        else:
            tag = "means overlapping -> early window infeasible"
        print(f"\nwindow K={K:<3} (first {K} epochs accumulated)")
        print(f"  CNN-euclid  {me:6.3f}±{(max(e) - min(e)) / 2:.3f}   "
              f"[{', '.join(f'{x:.2f}' for x in sorted(e))}]")
        print(f"  CNN-cosine  {mc:6.3f}±{(max(c) - min(c)) / 2:.3f}   "
              f"[{', '.join(f'{x:.2f}' for x in sorted(c))}]")
        print(f"  -> gap={diff:.3f} ({rel * 100:.1f}%) max spread={spread:.3f}  "
              f"min(euclid)={min(e):.3f}  max(cosine)={max(c):.3f}  -> {tag}")

    print("\n" + "=" * 96)
    print("Consistency across the four probe sets at K=10 (secondary check)")
    print("=" * 96)
    for pk in PROBE_KEYS:
        e = [r["curve"][pk][9] for r in rows
             if r["cell"] == "CNN-euclid" and len(r["curve"][pk]) >= 10]
        c = [r["curve"][pk][9] for r in rows
             if r["cell"] == "CNN-cosine" and len(r["curve"][pk]) >= 10]
        if e and c:
            gap = min(e) - max(c)
            print(f"  {pk:<11} euclid={sum(e) / len(e):6.3f}  "
                  f"cosine={sum(c) / len(c):6.3f}  "
                  f"min/max={min(e):.3f}/{max(c):.3f}  "
                  f"-> {'zero overlap' if gap > 0 else 'overlap'}")

    print("\n" + "=" * 96)
    print("Cross-check: early-window final values vs main sensitivity record (same-seed deterministic rerun, should be bit-for-bit identical)")
    print("=" * 96)
    main_rows = {(r["cell"], r["s"]): r for r in load_rows()}
    n_ok = n_bad = n_missing = 0
    for r in rows:
        m = main_rows.get((r["cell"], r["s"]))
        if not m:
            n_missing += 1
            continue
        dv = abs(r["curve"]["P1_te512"][-1] - m["l_funcs"]["P1_te512"])
        if dv < 1e-9:
            n_ok += 1
        else:
            n_bad += 1
            print(f"  [mismatch] {r['cell']} s{r['s']}: "
                  f"early {r['curve']['P1_te512'][-1]:.6f} vs "
                  f"main {m['l_funcs']['P1_te512']:.6f}")
    print(f"  match {n_ok}; mismatch {n_bad}; main record missing {n_missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell")
    ap.add_argument("--seeds", type=int, default=3,
                    help="seeds per cell (default 3; CP3 rho ground truth only exists for s0-s2)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--early", action="store_true",
                    help="record per-epoch accumulated L_func curves to cp3_dyn_early.jsonl during training")
    ap.add_argument("--early-report", action="store_true",
                    help="analyze early-window (first K epochs accumulated L_func) separability")
    args = ap.parse_args()
    jl = EARLY_JSONL if args.early else JSONL
    if args.early_report:
        _redirect_log(os.path.join(_DATA, "cp3_dyn_early_report.txt"))
        early_report()
        _LOG.flush()
        return
    if args.report:
        _redirect_log(os.path.join(_DATA, "cp3_dyn_sens_report.txt"))
        report()
        _LOG.flush()
        return
    if args.all:
        done = {(r["cell"], r["s"]) for r in load_rows(jl)}
        for c_idx, c in enumerate(CELLS):
            for s in range(args.seeds):
                if (c["name"], s) in done:
                    print(f"[skip] {c['name']} s{s} (exists)", flush=True)
                    continue
                run_and_log(c, c_idx, s, jl)
        return
    if args.cell:
        c_idx = [c["name"] for c in CELLS].index(args.cell)
        done = {(r["cell"], r["s"]) for r in load_rows(jl)}
        for s in range(args.seeds):
            if (args.cell, s) in done:
                print(f"[skip] {args.cell} s{s} (exists)", flush=True)
                continue
            run_and_log(CELLS[c_idx], c_idx, s, jl)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        _LOG.flush()
        sys.exit(1)          # no longer raise: re-raising after print_exc would print the same traceback twice
    _LOG.flush()
