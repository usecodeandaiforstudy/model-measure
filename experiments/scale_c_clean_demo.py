"""SCALE-C CLEAN: the class-count axis with the wall's CEILING HELD FIXED (the decisive measurement).

Why a third run. Round 1 (reference with no bottleneck) could not locate a wall at all for C >= 27; round 2
(reference = same-family VQ with K_t = 8C) located walls but the ceiling K_t scaled WITH C, so the fitted
wall exponent alpha = 2.018 absorbed the reference's scaling rather than the task's (class-count-axis 6.2).
This run holds the ceiling fixed: **K_t = 2048 for every C** (log 2048 = 7.62 nats >= ln 1000 = 6.91, so
the reference is not binding anywhere in the sweep) and widens the student grid up to K_t with ratio 2 at
the top, so the wall can be located wherever it actually is.

Performance fix (measured, not assumed). Round 2 paid a forward pass through the 2048/8192-code teacher at
EVERY student step (cdist 128 x K_t), which took a unit from 5.5 to 18 minutes. Here the teacher's logits
are materialised ONCE per instance (fp16 cache) and the student loop reads them -- the loss is byte-for-byte
the package's (CE + KL(T=3) * T^2 + beta * commit, with the same dead-code restart), and the replica is
verified numerically against capacity_lens.training.train_one_epoch in --selftest.

Pre-registration (2026-09-30, before the run; criteria S1-S5 are the SAME functions as SCALE-C, imported)
-----------------------------------------------------------------------------------------------
(1) Instances: m in {2,3,4,6,10} -> C in {8,27,64,216,1000}; lattice task of scale_c_demo.make_lattice
    (spacing 3.0, sigma 1.0, 150 samples/class, 80/20 split, seed 0).
(2) Models: GridMLP (3->64 -> bottleneck(64) -> 64->C). Reference: VQ(K_t = 2048), 60 ep, Adam 1e-2, CE.
    Students: VQ(K) for K in [2, 8, 32, 128, 512, 1024, 2048], same budget, KD from the cached targets.
(3) Readouts: eps -> R_IB -> K_min -> excluded set; per K: relative gap -> measured wall (rel_tol 0.10)
    -> looseness. Same conventions as SCALE-C.
(4) Criteria (scale_verdict, unchanged): S1 informative-eps gate; S2 wall scaling alpha (descriptive);
    S3 looseness non-increasing in C AND within 0.5 nats of alpha/(1-eps) at the largest C; S4 zero false
    exclusions at every C; S5 the exclusion's reach grows with C. Verdict: SUPPORT if S3 and S4 and S5.
(5) Prediction for this design (recorded before the data): with the ceiling fixed, alpha now measures the
    TASK's wall scaling; round 2's alpha = 2.018 was an upper bound contaminated by K_t = 8C, so the clean
    alpha should come out LOWER unless the task's wall genuinely grows quadratically in C.
(6) Relay: checkpoints and target caches under experiments/data/scale_c_clean/<fingerprint>/.
(7) Iron rules: criteria before data; pure functions unit-testable; English-only output; no downloads.
-----------------------------------------------------------------------------------------------
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__)))
from capacity_lens import (VQBottleneck, train_one_epoch, evaluate, task_info_fano, DistillCapacity,
                           excluded_set, coverage, false_exclusions, measured_wall, looseness)
from scale_c_demo import make_lattice, GridMLP, class_count, scale_verdict

M_GRID = [2, 3, 4, 6, 10]
K_GRID = [2, 8, 32, 128, 512, 1024, 2048]      # ratio 4, then ratio 2 up to the fixed ceiling
K_TEACHER = 2048                               # FIXED for every C (the point of this run)
EPOCHS = 60
LR = 1e-2
SIGMA = 1.0
N_PER_CLASS = 150
SEED = 0
T_KD = 3.0
BETA = 0.25

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
JSONL = os.path.join(_DATA, "scale_c_clean.jsonl")
LOG = os.path.join(_DATA, "scale_c_clean.log")
REPORT = os.path.join(_DATA, "scale_c_clean_report.txt")
CKPT_ROOT = os.path.join(_DATA, "scale_c_clean")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def _setup_logging():
    global _LOG
    _LOG = open(LOG, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _LOG)
    sys.stderr = _Tee(sys.__stderr__, _LOG)


def jsonl_append(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Cached-target student loop (an exact replica of the package's KD objective)
# ---------------------------------------------------------------------------

def train_one_epoch_cached(model, loader, optimizer, device="cpu"):
    """Line-for-line the package's objective, with the teacher's logits supplied by the loader instead of a
    per-step teacher forward: CE + KL(log_softmax(out/T) || softmax(t/T)) * T^2 + beta * commit, plus the
    same dead-code restart. Verified against train_one_epoch(distill_from=...) in run_selftest()."""
    model.train()
    total = 0.0
    for x, y, t_out in loader:
        x, y, t_out = x.to(device), y.to(device), t_out.to(device)
        out, _ = model(x)
        ce = F.cross_entropy(out, y)
        soft = F.kl_div(F.log_softmax(out / T_KD, dim=1), F.softmax(t_out / T_KD, dim=1),
                        reduction="batchmean") * (T_KD * T_KD)
        loss = ce + soft
        for m in model.modules():
            if isinstance(m, VQBottleneck):
                loss = loss + BETA * m.commit_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for m in model.modules():
            if isinstance(m, VQBottleneck):
                t = m.last_input()
                if t is not None:
                    m.update_usage_and_restart(t)
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def teacher_logits(model, loader, device):
    """Materialise the frozen reference's logits once (fp16: enough for soft targets, half the memory)."""
    outs = []
    for x, _ in loader:
        out, _ = model(x.to(device))
        outs.append(out.half().cpu())
    return torch.cat(outs)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_signature(m_grid, k_grid, epochs, k_teacher):
    raw = "|".join([f"M{','.join(map(str, m_grid))}", f"K{','.join(map(str, k_grid))}", f"ep{epochs}",
                    f"kt{k_teacher}", f"sig{SIGMA}", f"spc{N_PER_CLASS}", f"seed{SEED}", "lattice-clean"])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]


def loaders_for(m, seed=SEED, batch_size=128):
    (x_tr, y_tr), (x_te, y_te) = make_lattice(m, sigma=SIGMA, seed=seed)
    return (DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(TensorDataset(x_te, y_te), batch_size=512, shuffle=False, num_workers=0),
            (x_tr, y_tr), (x_te, y_te))


def run_instance(m, device, resume, ckpt_dir, epochs=EPOCHS):
    C = class_count(m)
    tr, te, (x_tr, y_tr), (x_te, y_te) = loaders_for(m)
    t0 = time.time()
    # 1) reference with the FIXED ceiling
    path_t = os.path.join(ckpt_dir, f"teacher_C{C}.pt")
    if resume and os.path.exists(path_t):
        acc_t = float(torch.load(path_t, map_location="cpu", weights_only=True)["acc"])
    else:
        torch.manual_seed(70000 + C)
        ref = GridMLP(C, K_TEACHER).to(device)
        opt = torch.optim.Adam(ref.parameters(), lr=LR)
        for _ in range(epochs):
            train_one_epoch(ref, tr, opt, device=device)
        acc_t, _ = evaluate(ref, te, device=device)
        torch.save({"acc": acc_t, "state_dict": ref.state_dict()}, path_t)
    ref = GridMLP(C, K_TEACHER).to(device)
    ref.load_state_dict(torch.load(path_t, map_location=device, weights_only=True)["state_dict"])
    ref.eval()
    t_ref = time.time() - t0
    # 2) materialise the soft targets once per instance
    path_c = os.path.join(ckpt_dir, f"targets_C{C}.pt")
    if resume and os.path.exists(path_c):
        cache = torch.load(path_c, map_location="cpu", weights_only=True)
    else:
        cache = dict(tr=teacher_logits(ref, tr, device), te=teacher_logits(ref, te, device))
        torch.save(cache, path_c)
    tr_c = DataLoader(TensorDataset(x_tr, y_tr, cache["tr"].float()),
                      batch_size=128, shuffle=True, num_workers=0)
    t_cache = time.time() - t0 - t_ref
    # 3) students
    accs = {}
    for K in K_GRID:
        path_s = os.path.join(ckpt_dir, f"student_C{C}_K{K}.pt")
        if resume and os.path.exists(path_s):
            accs[K] = float(torch.load(path_s, map_location="cpu", weights_only=True)["acc"])
            continue
        torch.manual_seed(71000 + C + K)
        stu = GridMLP(C, K).to(device)
        opt = torch.optim.Adam(stu.parameters(), lr=LR)
        for _ in range(epochs):
            train_one_epoch_cached(stu, tr_c, opt, device=device)
        acc, _ = evaluate(stu, te, device=device)
        torch.save({"acc": acc, "state_dict": stu.state_dict()}, path_s)
        accs[K] = acc
    # 4) readouts (identical conventions to SCALE-C)
    eps = 1.0 - acc_t
    rib = task_info_fano(eps, C)
    k_min = DistillCapacity().min_codebook_size(rib)
    rel = {K: (acc_t - a) / max(acc_t, 1e-12) for K, a in accs.items()}
    fails = sorted(K for K in accs if rel[K] >= 0.50)
    succeeds = sorted(K for K in accs if rel[K] <= 0.10)
    excl = excluded_set(rib, sorted(accs))
    k_wall, _ = measured_wall(sorted(accs), accs, acc_t, rel_tol=0.10)
    lo = looseness(rib, k_wall)
    row = dict(c=C, m=m, seed=SEED, acc_t=acc_t, eps=eps, rib=rib, k_min=k_min, accs=accs, rel=rel,
               fails=fails, succeeds=succeeds, excluded=excl, k_wall=k_wall, looseness=lo,
               s1=(0.02 <= eps <= 0.60), k_teacher=K_TEACHER,
               false_excl=false_exclusions(excl, succeeds), coverage=coverage(excl, fails),
               sigma=SIGMA, t_ref=t_ref, t_cache=t_cache, t_total=time.time() - t0)
    jsonl_append(JSONL, row)
    print(f"  C={C:5d} (m={m:2d}): eps={eps:.3f} R_IB={rib:.2f} K_min={k_min:4d} excl={excl} "
          f"K_wall={k_wall} looseness={None if lo is None else round(lo, 2)} false={row['false_excl']} "
          f"[ref {t_ref:.0f}s + cache {t_cache:.0f}s + students {time.time() - t0 - t_ref - t_cache:.0f}s]")
    print("      rel gaps: " + "  ".join(f"K{K}:{rel[K]:.2f}" for K in sorted(accs)))
    return row


def run_clean(m_grid, resume, device, epochs=EPOCHS):
    fp = run_signature(m_grid, K_GRID, epochs, K_TEACHER)
    ckpt_dir = os.path.join(CKPT_ROOT, fp)
    os.makedirs(ckpt_dir, exist_ok=True)
    print("=" * 92)
    print("[SCALE-C CLEAN] class-count axis with the wall's CEILING HELD FIXED (K_t = 2048 for every C)")
    print("=" * 92)
    print(f"  [checkpoints] {'relay mode (cache hit -> skip)' if resume else 'recompute mode'}  {ckpt_dir}")
    print(f"  [design] lattice family, sigma {SIGMA}, {N_PER_CLASS} samples/class, seed {SEED}; "
          f"reference VQ({K_TEACHER}) fixed; students K in {K_GRID}; {epochs} ep, Adam {LR}")
    print(f"  [method] teacher logits cached per instance (exact replica of the package KD objective)")
    rows = [run_instance(m, device, resume, ckpt_dir, epochs) for m in m_grid]
    print("\n  --- SCALE-C CLEAN verdict ---")
    verdict, detail = scale_verdict(rows)
    for line in detail:
        print(line)
    print(f"  [verdict] {verdict}")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("[SCALE-C CLEAN] fixed ceiling\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for line in detail:
            f.write(line + "\n")
        f.write("verdict: " + verdict + "\n")
    print(f"  [report] {REPORT}")
    return rows, verdict


def run_budget_probe(m, k_list, epochs, device, resume=True, fp="a8772659"):
    """Locate the C=1000 wall with a LONGER student budget (the follow-up recorded as item 7).

    Round 3 left C=1000 unlocated: at K=2048 the 60-epoch student was still 26% short of the reference.
    WALL-SATURATION then showed that the students' retention deficit is partly budget-limited (3x the epochs
    halved it at C=216, sigma=0.5). This probe applies that lever to the unlocated instance, reusing round 3's
    cached teacher and cached soft targets, so the protocol is identical and only the student budget changes.

    Pre-registered criteria (2026-10-05, before the run):
      W1  the gap at the largest probed K falls below 0.10 -> the C=1000 wall is LOCATED (<= that K) and the
          round-3 "not located" record is superseded;
      W2  the gap falls by roughly the C=216 factor (to about 0.14, i.e. an improvement ratio of 0.45-0.65) but
          stays above 0.10 -> still not located, and the budget-scaling hypothesis is CONFIRMED;
      W3  the gap improves by less than 20% -> at C=1000 the deficit is NOT budget-limited (a different regime
          from C=216) and the unlocated record stands for a reason of its own.
    Reported as measured whichever holds; no criterion is adjusted afterwards.
    """
    from torch.utils.data import DataLoader, TensorDataset
    from capacity_lens import evaluate

    C = class_count(m)
    ckpt_dir = os.path.join(CKPT_ROOT, fp)
    path_t = os.path.join(ckpt_dir, f"teacher_C{C}.pt")
    path_c = os.path.join(ckpt_dir, f"targets_C{C}.pt")
    if not (os.path.exists(path_t) and os.path.exists(path_c)):
        raise SystemExit(f"[SCALE-C BUDGET] round-3 cache missing for C={C} under {ckpt_dir}")
    acc_t = float(torch.load(path_t, map_location="cpu", weights_only=True)["acc"])
    cache = torch.load(path_c, map_location="cpu", weights_only=True)
    (x_tr, y_tr), (x_te, y_te) = make_lattice(m, n_per_class=N_PER_CLASS, sigma=SIGMA, seed=SEED)
    te = DataLoader(TensorDataset(x_te, y_te), batch_size=512, shuffle=False, num_workers=0)
    tr_c = DataLoader(TensorDataset(x_tr, y_tr, cache["tr"].float()), batch_size=128, shuffle=True,
                      num_workers=0)
    rows_60 = {}
    if os.path.exists(JSONL):
        # round 3's rows do NOT carry an `epochs` field, and the C=1000 row is precisely the one with
        # k_wall = None, so the comparison row is selected by class count alone; when several runs left rows
        # for the same C the longest one wins (t_total = the official full run). Two wrong versions of this
        # selector silently produced a wrong verdict before it was fixed.
        best = None
        for line in open(JSONL, encoding="utf-8"):
            r = json.loads(line)
            if r.get("c") == C and "rel" in r:
                if best is None or r.get("t_total", 0) > best.get("t_total", 0):
                    best = r
        rows_60 = best or {}
    print("=" * 100)
    print(f"[SCALE-C BUDGET] C={C}: student budget {EPOCHS} -> {epochs} epochs, K in {k_list} "
          f"(teacher and soft targets reused from round 3, acc_t={acc_t:.3f})")
    print("=" * 100)
    eps = 1.0 - acc_t
    rib = task_info_fano(eps, C)
    print(f"  reference: acc={acc_t:.3f} (eps={eps:.3f}), R_IB={rib:.2f}, "
          f"K_min={DistillCapacity().min_codebook_size(rib)}")
    if rows_60:
        print(f"  the 60-epoch run recorded: K_wall={rows_60.get('k_wall')}, rel gaps "
              f"{({int(k): round(v, 3) for k, v in rows_60.get('rel', {}).items()})}")
    out = []
    for K in k_list:
        path_s = os.path.join(ckpt_dir, f"student_C{C}_K{K}_ep{epochs}.pt")
        if resume and os.path.exists(path_s):
            acc = float(torch.load(path_s, map_location="cpu", weights_only=True)["acc"])
        else:
            torch.manual_seed(71000 + C + K)
            stu = GridMLP(C, K).to(device)
            opt = torch.optim.Adam(stu.parameters(), lr=LR)
            for _ in range(epochs):
                train_one_epoch_cached(stu, tr_c, opt, device=device)
            acc, _ = evaluate(stu, te, device=device)
            torch.save({"acc": acc, "state_dict": stu.state_dict()}, path_s)
        rel = (acc_t - acc) / acc_t
        rel_60 = rows_60.get("rel", {}).get(str(K)) if rows_60 else None
        out.append(dict(arm="budget_c1000", c=C, m=m, k=K, epochs=epochs, acc_t=acc_t, acc=acc, rel=rel,
                        rel_60=rel_60, sigma=SIGMA, rib=rib,
                        k_min=DistillCapacity().min_codebook_size(rib)))
        ratio = (rel / rel_60) if rel_60 else None
        print(f"  K={K:5d}: acc={acc:.3f} -> gap {rel:+.3f}"
              + (f" (was {rel_60:+.3f} at {EPOCHS} epochs, ratio {ratio:.2f})" if rel_60 else ""))
    located = [r for r in out if r["rel"] <= 0.10]
    if located:
        v = (f"W1: the wall is LOCATED at K={min(r['k'] for r in located)} with {epochs} epochs "
             f"(the round-3 'not located' record is superseded)")
    else:
        ratios = [r["rel"] / r["rel_60"] for r in out if r["rel_60"]]
        best = min(ratios) if ratios else None
        if best is not None and 0.45 <= best <= 0.65:
            v = (f"W2: still NOT located, and the budget-scaling hypothesis is confirmed "
                 f"(improvement ratio {best:.2f}; the C=216 value was 0.55)")
        elif best is not None and best < 0.80:
            v = f"W2': still NOT located; the gap improved by {(1 - best):.0%} (outside the predicted band)"
        else:
            v = ("W3: the gap did not respond to the budget -> at C=1000 the deficit is NOT budget-limited "
                 "(a different regime from C=216) and the unlocated record stands for its own reason")
    print(f"\n  [verdict] {v}")
    with open(JSONL, "a", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(REPORT, "a" if os.path.exists(REPORT) else "w", encoding="utf-8") as f:
        f.write(f"budget probe C={C} epochs={epochs}: {v}\n")
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out, v


def verdict_only():
    if not os.path.exists(JSONL):
        raise SystemExit("[SCALE-C CLEAN] nothing to re-verdict")
    rows_by_key = {}
    skipped = 0
    for line in open(JSONL, encoding="utf-8"):
        r = json.loads(line)
        if "eps" not in r:                       # e.g. the budget-probe rows: a different row shape
            skipped += 1
            continue
        rows_by_key[(r["c"], r.get("seed", 0))] = r   # round-3 rows carry no seed key
    rows = [rows_by_key[k] for k in sorted(rows_by_key)]
    print("[SCALE-C CLEAN offline re-verdict]"
          + (f"  ({skipped} non-scale row(s) skipped: a different row shape)" if skipped else ""))
    for r in rows:
        print(f"  C={r['c']:5d}: eps={r['eps']:.3f} R_IB={r['rib']:.2f} K_min={r['k_min']:4d} "
              f"excl={r['excluded']} K_wall={r['k_wall']} "
              f"looseness={None if r['looseness'] is None else round(r['looseness'], 2)}")
    verdict, detail = scale_verdict(rows)
    for line in detail:
        print(line)
    print(f"  [verdict] {verdict}")


def run_selftest():
    ok = []
    # (a) the cached-target replica must match the package's KD loop numerically
    device = "cpu"
    (x_tr, y_tr), (x_te, y_te) = make_lattice(2, n_per_class=6, sigma=SIGMA, seed=0)
    ref = GridMLP(8, K_TEACHER)
    tr_plain = DataLoader(TensorDataset(x_tr, y_tr), batch_size=16, shuffle=False)
    tr_cached = DataLoader(TensorDataset(x_tr, y_tr, teacher_logits(ref, tr_plain, device).float()),
                           batch_size=16, shuffle=False)
    torch.manual_seed(0)
    s1 = GridMLP(8, 8)
    o1 = torch.optim.Adam(s1.parameters(), lr=LR)
    l_plain = train_one_epoch(s1, tr_plain, o1, device=device, distill_from=ref, T_kd=T_KD, beta=BETA)
    torch.manual_seed(0)
    s2 = GridMLP(8, 8)
    o2 = torch.optim.Adam(s2.parameters(), lr=LR)
    l_cached = train_one_epoch_cached(s2, tr_cached, o2, device=device)
    ok.append((f"cached-target KD matches the package loop ({l_plain:.4f} vs {l_cached:.4f})",
               abs(l_plain - l_cached) < 1e-4))
    # (b) the fixed ceiling must not bind: log K_t >= ln(max C)
    ok.append(("K_t = 2048 is not binding for C <= 1000",
               math.log(K_TEACHER) >= math.log(class_count(10))))
    # (c) the grid reaches the ceiling
    ok.append(("the student grid reaches K_t", max(K_GRID) == K_TEACHER))
    print("[SCALE-C CLEAN selftest] replica + design checks:")
    all_ok = True
    for name, passed in ok:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        all_ok = all_ok and passed
    print(f"  [selftest verdict] {'all PASS' if all_ok else 'FAILURES present (code defect)'}")
    return all_ok


def main():
    ap = argparse.ArgumentParser(description="SCALE-C CLEAN: fixed-ceiling class-count sweep")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--quick", action="store_true", help="C=8 only, few epochs (smoke)")
    ap.add_argument("--fresh", action="store_true", help="recompute (default: relay mode)")
    ap.add_argument("--selftest", action="store_true", help="replica + design checks (no sweep)")
    ap.add_argument("--verdict-only", action="store_true", help="re-verdict from the jsonl (no training)")
    ap.add_argument("--budget-probe", action="store_true",
                    help="locate the C=1000 wall with a longer student budget (item 7 follow-up)")
    ap.add_argument("--probe-m", type=int, default=10, help="lattice parameter for the budget probe (10 -> C=1000)")
    ap.add_argument("--probe-k", type=int, nargs="+", default=[2048])
    ap.add_argument("--probe-epochs", type=int, default=180)
    args = ap.parse_args()
    if args.selftest:
        run_selftest()
        return
    if args.verdict_only:
        verdict_only()
        return
    if args.budget_probe:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device: {device}")
        run_budget_probe(args.probe_m, args.probe_k, args.probe_epochs, device, resume=not args.fresh)
        return
    _setup_logging()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if args.quick:
        run_clean([2], not args.fresh, device, epochs=5)
    else:
        run_clean(M_GRID, not args.fresh, device, epochs=args.epochs)
    print("\nDone.")


if __name__ == "__main__":
    main()
