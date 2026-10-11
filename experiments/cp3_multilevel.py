# -*- coding: utf-8 -*-
"""CP3 supplementary test ③: multi-level dose test (axis A: teacher codebook K_t) — is early-window L_func order-preserving with respect to rho?

Motivation (after CP3 supplementary test ② + the robustness triple):
  L_func (the accumulated prediction flips) has per-seed zero overlap at two rho levels (CNN 2.00 vs 3.00),
  but "order preservation" is unproven: if the reward is not order-preserving with respect to closer configurations, the optimizer would be misled.
  This experiment creates >=3 rho levels along the **task-side dose axis** (teacher codebook K_t),
  testing the ranking consistency of the early-window L_func.

Design:
  Cell: CNN-euclid (the main cell where L_func is already validated).
  Levels: K_t in {128, 256, 512}.
    - 256 reuses the archive: rho=3.00 (CP3), teacher acc/H(Z_t) and the early-window L_func curve
      (cp3_dyn_early.jsonl, s0-s2), not recomputed;
    - 128/512 are newly computed. The teacher seed formula is bit-for-bit identical to CP3's (14000+100*2+s,
      c_idx=2 = CNN-euclid); with the same data seed -> the only variable of the teacher relative to the archive = codebook size
      (paired design). The student seed 15000+200+100*s+K is on the same basis.
  Per level per seed: teacher training (G0 double gate + 4-probe-set L_func curve)
    -> R_IB(Fano) -> K_min -> student distillation grid (CE + KL(T=3), same as CP3)
    -> plateau check -> K* -> rho = K*/K_min.
  The K grid is truncated/extended with the level (the upper bound of K* is constrained by the teacher's effective codebook; the grid must cover up to K_t):
    K_t=128 -> [8, 32, 48, 64, 96, 128]
    K_t=512 -> [8, 32, 48, 64, 96, 128, 192, 256, 384, 512]

Pre-registration (2026-09-23, criteria prior to data; this docstring is the pre-registration record)
--------------------------------------------------------------------
(1) Mechanism prediction: K_min = ceil(exp(Fano(1-acc, C=64))) depends only on the teacher error.
    When all three levels pass G0 (acc~1), K_min~64 is constant, and all variation in rho comes from K*.

(2) Upper-bound argument for K*(K_t=128): once the student codebook exceeds the teacher's effective codebook, there is no extra information
    to match, and the cliff must fall below H_eff(K_t=128) -> rho(128) < rho(256) is almost inevitable
    (this half does not constitute a test); the direction of rho(512) relative to rho(256) is not predicted (the real question).

(3) Main criterion (order preservation): the three levels' (rho, L_func@early-window K=10) rankings agree -> order preservation holds,
    and the reward can be used for regression-type optimization. Adjacent levels are judged separable by "difference > max half-range" (consistent with the
    previous convention). For level pairs with equal rho, L_func is also expected to be equal.

(4) Discriminative outcomes (recorded honestly regardless of outcome):
    - rho changes while L_func does not follow -> L_func does not track rho (a key negative result; the reward
      is downgraded to a binary discriminator);
    - rho is flat (K* sticks to K_t, can only occur for 256 vs 512) while L_func changes -> same as above.
    Note the tension with CP2: CP2 found on MLP that K* "sticks" to the dose (96-128 ≈ 2xK_min).
    If on CNN K* also sticks while L_func still changes, it means L_func follows the absolute task difficulty
    (K_min or the training dynamics itself) rather than the surplus ratio.

(5) G0 double gate (acc>=0.95 and H(Z_t)>=0.8*ln(K_t), the gate using each level's own K_t) or a (level, seed) whose plateau
    is not established is honestly recorded as invalid, with no threshold retro-application.
    Budget upshift contingency (prior to any decision data): the K_t=512 codebook is larger, and 40ep may under-converge
    —— if the 512-level teacher does not pass G0, one budget-upshift retry is allowed (80ep, lr unchanged);
    the upshift decision must precede the generation of any student data at that level, and the 128 level is not upshifted
    (to keep isomorphic with the CP3 calibration budget).
    Revision (2026-09-23, after all seeds at the 128 level failed G0-acc at 40ep and before any of its student
    data was generated): a one-time same-specification upshift retry for the 128 level is approved (80ep, lr unchanged).
    Rationale: the 128 teacher's L_func curve shows it is still undergoing violent reorganization at ep40 (the last 10 epochs' increment
    1.6-2.8, versus freezing at ep10 for the 256 level), which is budget insufficiency rather than structural failure; the revision follows
    the "recalibration precedes decision data" rule (128 has zero student data). If 80ep still does not pass the gate,
    128 is permanently recorded as invalid, with no further upshift.

(6) Iron rule: print uses only GBK-encodable characters; criteria prior to data; negative results recorded honestly;
    checkpoints are persisted per cell, so an interruption loses at most one cell.

Usage:
  python experiments/cp3_multilevel.py --level 128
  python experiments/cp3_multilevel.py --level 512
  python experiments/cp3_multilevel.py --report
"""
import argparse
import json
import math
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data")
sys.path.append(os.path.join(_HERE, ".."))
sys.path.append(_HERE)

import torch
from torch.utils.data import DataLoader

from capacity_lens import train_one_epoch, evaluate, entropy_of, task_info_fano
from mnist_demo import distill_gap_k_star, distill_plateau_ok, g0_gate
from cp3_demo import make_data, build_model, cell_budget, CALIBRATED_K

JSONL = os.path.join(_DATA, "cp3_multilevel.jsonl")
LOG = os.path.join(_DATA, "cp3_multilevel.log")
CKPT_ROOT = os.path.join(_DATA, "cp3_multilevel")
EARLY_JSONL = os.path.join(_DATA, "cp3_dyn_early.jsonl")   # source of the K_t=256 L_func curve

N_PER_CLASS = 150          # consistent with the CP3 decision run
C_TASK = 64                # grid64 task class count (the Fano capacity term of K_min, independent of K_t)
CELL_IDX = 2               # the index of CNN-euclid in CP3 CELLS (the seed formula aligns with the archive)
RHO_256_ARCHIVE = 3.00     # the CP3-archived CNN-euclid rho (K_t=256)

LEVELS_NEW = [128, 512]    # levels needing new computation; 256 only reuses the archive
VALID_TEACHER_EPOCHS = {40, 80}   # calibration budget and the 512 upshift contingency (pre-registration (5));
# other epoch counts = smoke probes; their JSONL rows and checkpoints never participate in the formal results
PROBES = {
    "P1_te512": ("te", 0, 512),
    "P2_te512b": ("te", 512, 1024),
    "P3_te1024": ("te", 0, 1024),
    "P4_tr512": ("tr", 0, 512),
}
PROBE_KEYS = list(PROBES)


class _Tee:
    """stdout/stderr written to both the terminal and the log file (host capture is unreliable)."""

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


def grid_for(level):
    """The K grid is truncated/extended with the level (pure function)."""
    if level <= 128:
        return [k for k in CALIBRATED_K if k <= level]
    return list(CALIBRATED_K) + [384, 512]


def train_teacher_probes(level, s, epochs, lr, ckpt_dir, resume):
    """The CNN-euclid teacher with K_t=level: training + G0 readouts + 4-probe-set L_func curves.

    The seed formula is bit-for-bit identical to CP3's (14000+100*CELL_IDX+s); with the same data seed, the only variable
    relative to the CP3-archived teacher = codebook size (paired design).
    """
    torch.manual_seed(14000 + 100 * CELL_IDX + s)
    (xtr, ytr), (xte, yte) = make_data("cnn", N_PER_CLASS, s)
    train_loader = DataLoader(list(zip(xtr, ytr)), batch_size=128, shuffle=True)
    test_loader = DataLoader(list(zip(xte, yte)), batch_size=256)
    model = build_model("cnn", level, "euclid")
    path = os.path.join(ckpt_dir, f"kt{level}_teacher_s{s}.pt")
    if resume and os.path.exists(path):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if ck.get("epochs") != epochs:
            print(f"    seed {s}: [cache epochs mismatch {ck.get('epochs')}ep != {epochs}ep]"
                  f" retraining", flush=True)
        else:
            model.load_state_dict(ck["state_dict"])
            print(f"    seed {s}: [cache hit] teacher acc={ck['acc']:.3f}  "
                  f"H(Z_t)={ck['h_z']:.2f}", flush=True)
            return dict(acc=ck["acc"], h_z=ck["h_z"], l_funcs=ck["l_funcs"],
                        curve=ck["curve"], model=model, tr=train_loader,
                        te=test_loader)

    probes = build_probes(xtr, xte)
    preds = {k: probe_pred(model, v) for k, v in probes.items()}
    l_funcs = {k: 0.0 for k in probes}
    curves = {k: [] for k in probes}
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for e in range(epochs):
        train_one_epoch(model, train_loader, opt)
        for k, v in probes.items():
            p = probe_pred(model, v)
            l_funcs[k] += float((p != preds[k]).float().mean())
            preds[k] = p
            curves[k].append(l_funcs[k])
        if (e + 1) % max(1, epochs // 4) == 0 or e == 0:
            print(f"      ep{e + 1}/{epochs}  P1={l_funcs['P1_te512']:.2f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    acc, idx = evaluate(model, test_loader)
    h_z = entropy_of(idx[0], level)
    torch.save({"state_dict": model.state_dict(), "acc": acc, "h_z": h_z,
                "l_funcs": l_funcs, "curve": curves, "epochs": epochs}, path)
    return dict(acc=acc, h_z=h_z, l_funcs=l_funcs, curve=curves,
                model=model, tr=train_loader, te=test_loader)


def run_students(level, s, teacher, tr, te, epochs, lr, ckpt_dir, resume,
                 teacher_epochs):
    """The distillation grid for K_t=level; returns (k_list, {K: acc}). The seed formula is consistent with CP3's.

    Checkpoints carry a tep label (the epoch count of the distillation-source teacher): student caches produced by
    smoke probes or upshift retries will not be misused by the formal run.
    """
    k_list = grid_for(level)
    accs = {}
    for K in k_list:
        torch.manual_seed(15000 + 100 * CELL_IDX + 100 * s + K)
        student = build_model("cnn", K, "euclid")
        path = os.path.join(ckpt_dir, f"kt{level}_student_s{s}_K{K}.pt")
        if resume and os.path.exists(path):
            ck = torch.load(path, map_location="cpu", weights_only=True)
            if ck.get("tep") == teacher_epochs:
                accs[K] = float(ck["acc"])
                print(f"      K={K}: [cache hit] acc={accs[K]:.3f}", flush=True)
                continue
            print(f"      K={K}: [cache tep={ck.get('tep')} != {teacher_epochs}]"
                  f" retraining", flush=True)
        opt = torch.optim.Adam(student.parameters(), lr=lr)
        for _ in range(epochs):
            train_one_epoch(student, tr, opt, distill_from=teacher)
        acc_s, _ = evaluate(student, te)
        accs[K] = acc_s
        torch.save({"acc": acc_s, "tep": teacher_epochs}, path)
    return k_list, accs


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
            if r.get("epochs") not in VALID_TEACHER_EPOCHS:
                continue          # smoke-probe rows do not participate in the formal results or the resume decision
            last[(r["level"], r["s"])] = r
    return [last[k] for k in sorted(last)]


def run_level(level, seeds, resume, epochs_override=None):
    budget = cell_budget("cnn", quick=False)
    epochs = epochs_override or budget["epochs"]
    smoke = epochs_override == 2          # smoke probe: only exercises the pipeline; its artifacts do not participate in the formal results
    if not smoke:
        if epochs not in VALID_TEACHER_EPOCHS:
            raise SystemExit(f"[REJECTED] teacher epochs {epochs} not in valid budget tiers "
                             f"{sorted(VALID_TEACHER_EPOCHS)} (40=calibration, 80=512 upshift contingency)")
        if epochs_override and level not in (128, 512):
            raise SystemExit("[REJECTED] upshift contingency only for K_t=128/512"
                             " (pre-registration (5) and its 2026-09-23 revision)")
    lr = budget["lr"]
    # for the smoke probe (--epochs 2) the student also uses a short budget to exercise the pipeline; the formal run is always 40ep;
    # the 512 upshift (--epochs 80) only upshifts the teacher, while the student stays at 40ep (consistent with the CP3 protocol)
    student_epochs = (budget["epochs"] if not epochs_override
                      else min(epochs, budget["epochs"]))
    ckpt_dir = os.path.join(CKPT_ROOT, f"kt{level}")
    os.makedirs(ckpt_dir, exist_ok=True)
    print("=" * 76)
    print(f"[multi-level] K_t={level}  CNN-euclid  {epochs}ep lr={lr}  "
          f"{N_PER_CLASS}/class  grid={grid_for(level)}")
    print(f"  [checkpoint] {'resume mode' if resume else 'recompute mode'}  {ckpt_dir}")
    print("=" * 76)
    # seeds that fail G0 at the calibration budget (40ep) do not take a slot — the upshift retry (80ep) is the final chance;
    # if still failing after 80ep it is final (pre-registration (5) revision: each level is upshifted only once)
    done = set()
    for r in load_rows():
        if r["g0"] or r["epochs"] == 80:
            done.add((r["level"], r["s"]))
    for s in range(seeds):
        if (level, s) in done:
            print(f"  [skip] s{s} (already done)", flush=True)
            continue
        print(f"  --- s{s} ---", flush=True)
        t = train_teacher_probes(level, s, epochs, lr, ckpt_dir, resume)
        g0, ok_acc, ok_cb = g0_gate(t["acc"], t["h_z"], level)
        tag = "PASS" if g0 else (f"FAIL acc{'PASS' if ok_acc else 'FAIL'}/"
                                  f"codebook{'PASS' if ok_cb else 'collapsed'}")
        r_ib = task_info_fano(1.0 - t["acc"], C_TASK)
        k_min = max(2, math.ceil(math.exp(r_ib)))     # same convention as cp3_demo
        print(f"    teacher acc={t['acc']:.4f} (G0 {tag})  H(Z_t)={t['h_z']:.2f}  "
              f"R_IB={r_ib:.2f} -> K_min={k_min}", flush=True)
        row = dict(level=level, s=s, k_t=level, epochs=epochs, acc=t["acc"],
                   h_z=t["h_z"], g0=g0, r_ib=r_ib, k_min=k_min,
                   l_funcs=t["l_funcs"], curve=t["curve"])
        if not g0:
            row.update(plateau=None, k_star=None, rho=None, note="G0 FAIL")
        else:
            k_list, accs = run_students(level, s, t["model"], t["tr"], t["te"],
                                        budget["epochs"], lr, ckpt_dir, resume,
                                        teacher_epochs=epochs)
            plateau = distill_plateau_ok(k_list, accs)
            k_star = distill_gap_k_star(t["acc"], k_list, accs) if plateau else None
            rho = (k_star / k_min) if k_star else None
            gaps = {K: t["acc"] - accs[K] for K in k_list}
            print("    gap g(K): " + "  ".join(f"{K}:{gaps[K]:.3f}"
                                              for K in k_list), flush=True)
            print(f"    plateau={'yes' if plateau else 'no'}  K*={k_star}  "
                  f"rho={rho if rho is None else round(rho, 3)}", flush=True)
            row.update(k_list=k_list, accs=accs, plateau=plateau,
                       k_star=k_star, rho=rho)
        del t["model"]
        with open(JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


def _early_lfunc_256(seeds=3, at=10):
    """Fetch the early-window L_func of K_t=256 (CNN-euclid) from cp3_dyn_early.jsonl."""
    if not os.path.exists(EARLY_JSONL):
        return []
    out = []
    with open(EARLY_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["cell"] != "CNN-euclid" or r["s"] >= seeds:
                continue
            c = r["curve"]["P1_te512"]
            if len(c) >= at:
                out.append(dict(s=r["s"], acc=r["acc"], h_z=r["h_z"],
                                curve=r["curve"], l_early=c[at - 1]))
    return out


def _stat(v):
    if not v:
        return "  [MISSING]"
    m = sum(v) / len(v)
    return f"{m:7.3f}±{(max(v) - min(v)) / 2:.3f}"


def report():
    rows = load_rows()
    print("=" * 96)
    print("[multi-level dose] early-window L_func order preservation w.r.t. rho (CNN-euclid, P1_te512)")
    print("  criterion (pre-registered): the three levels' (rho, L_func@K=10) rankings agree -> order preservation holds;")
    print("  rho changes while L_func does not follow (or reverses) -> L_func tracks something other than rho (key negative result)")
    print("=" * 96)
    levels = {}
    r256 = _early_lfunc_256()
    if r256:
        k_min256 = max(2, math.ceil(math.exp(
            task_info_fano(1.0 - r256[0]["acc"], C_TASK))))
        levels[256] = dict(rho=RHO_256_ARCHIVE, k_min=k_min256, rows=r256,
                           note="reuse CP3 archived rho; acc/H(Z_t)/L_func from cp3_dyn_early")
    for lv in LEVELS_NEW:
        rs = [r for r in rows if r["level"] == lv]
        if rs:
            g0s = [r["g0"] for r in rs]
            rhos = [r["rho"] for r in rs if r["rho"] is not None]
            levels[lv] = dict(rho=(sum(rhos) / len(rhos) if rhos else None),
                              k_min=rs[0]["k_min"], rows=rs,
                              note=f"G0 PASS {sum(g0s)}/{len(g0s)}; "
                                   f"K*={[r['k_star'] for r in rs]}")

    for lv in sorted(levels):
        d = levels[lv]
        print(f"\n--- K_t={lv}  ({d['note']}) ---")
        if lv == 256:
            for r in d["rows"]:
                print(f"  s{r['s']}: acc={r['acc']:.4f}  H(Z_t)={r['h_z']:.2f}"
                      f"  L_func@10={r['l_early']:.3f}")
            v5 = [r["curve"]["P1_te512"][4] for r in d["rows"]
                  if len(r["curve"]["P1_te512"]) >= 5]
            v10 = [r["l_early"] for r in d["rows"]]
        else:
            for r in d["rows"]:
                le = (f"{r['curve']['P1_te512'][9]:.3f}"
                      if len(r["curve"]["P1_te512"]) >= 10 else "  -")
                ks = f"{r['k_star']}" if r["k_star"] else "  -"
                rr = f"{r['rho']:.3f}" if r["rho"] else "  -"
                print(f"  s{r['s']}: acc={r['acc']:.4f}  H(Z_t)={r['h_z']:.2f}  "
                      f"G0={'PASS' if r['g0'] else 'FAIL'}  K_min={r['k_min']}  "
                      f"K*={ks}  rho={rr}  L_func@10={le}")
            v5 = [r["curve"]["P1_te512"][4] for r in d["rows"]
                  if len(r["curve"]["P1_te512"]) >= 5]
            v10 = [r["curve"]["P1_te512"][9] for r in d["rows"]
                   if len(r["curve"]["P1_te512"]) >= 10]
        d["v5"], d["v10"] = v5, v10
        print(f"  summary: rho={d['rho'] if d['rho'] is None else round(d['rho'], 3)}"
              f"  K_min={d['k_min']}  "
              f"L_func@5={_stat(v5)}  L_func@10={_stat(v10)}")

    print("\n" + "=" * 96)
    print("order-preservation verdict (main criterion, L_func@K=10, sorted by rho)")
    print("=" * 96)
    ok = [(lv, levels[lv]["rho"], levels[lv]["v10"]) for lv in sorted(levels)
          if levels[lv]["v10"] and levels[lv]["rho"] is not None]
    if len(ok) < 2:
        print("  [insufficient data, UNDECIDABLE]")
        return
    ok.sort(key=lambda x: x[1])
    names = [f"K_t={lv}(rho={rho:.2f})" for lv, rho, _ in ok]
    means = [sum(v) / len(v) for _, _, v in ok]
    print("  rho order: " + " < ".join(names))
    print("  L_func@10 order: " + "  ".join(f"{m:.3f}" for m in means))
    mono = all(means[i] < means[i + 1] for i in range(len(means) - 1))
    anti = all(means[i] > means[i + 1] for i in range(len(means) - 1))
    sep_all = True
    for i in range(len(ok) - 1):
        v1, v2 = ok[i][2], ok[i + 1][2]
        dmean = abs(sum(v2) / len(v2) - sum(v1) / len(v1))
        spread = max((max(v1) - min(v1)) / 2, (max(v2) - min(v2)) / 2)
        sep = dmean > spread
        sep_all = sep_all and sep
        print(f"  adjacent pair {ok[i][0]}->{ok[i + 1][0]}: diff={dmean:.3f}  "
              f"max half-range={spread:.3f}  -> {'separable' if sep else 'overlapping'}")
    rhos_sorted = [rho for _, rho, _ in ok]
    if len(set(rhos_sorted)) == 1:
        print("\n  [verdict] valid levels' rho is flat while L_func changes significantly"
              " -> pre-registered (4)-b: L_func tracks something other than rho (recorded honestly)")
        return
    if mono and sep_all:
        print("\n  [verdict] L_func is monotone in rho and adjacent levels are separable -> order preservation holds, reward usable for regression-type optimization")
    elif mono:
        print("\n  [verdict] monotone direction but adjacent overlap exists -> trend correct, discrimination needs multi-seed averaging")
    elif anti:
        print("\n  [verdict] reverse monotone -> L_func tracks something other than rho (key negative result, recorded honestly)")
    else:
        print("\n  [verdict] non-monotone -> not order-preserving (recorded honestly, check each level's G0/plateau validity)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, choices=LEVELS_NEW)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int,
                    help="override teacher epochs (only for the 512 upshift contingency, must precede student data)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        _redirect_log(os.path.join(_DATA, "cp3_multilevel_report.txt"))
        report()
        _LOG.flush()
        return
    if args.level:
        run_level(args.level, args.seeds, args.resume, args.epochs)
        return
    for lv in LEVELS_NEW:
        run_level(lv, args.seeds, args.resume)
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        _LOG.flush()
        sys.exit(1)
    _LOG.flush()
