"""TF-EXCLUSION: the exclusion pipeline end-to-end on a Transformer, and its STRUCTURAL boundary.

Two questions, both never tested in this project:
  Q1 (end-to-end): does the whole five-step pipeline (reference -> R_IB -> K_min -> student grid ->
     R1a/R1b exclusion -> wall -> looseness) run on a Transformer at all?
  Q2 (Transformer-specific boundary): a residual Transformer has TWO bottleneck locations that behave
     completely differently -- on the pooled serial path (binding: supply = log K) and inside the FFN
     branch, where the residual stream BYPASSES it (T2 found +0.519; P4 found the min bound fails under a
     bypass). So does the exclusion verdict still mean anything when the VQ sits in the FFN branch?

Usage:
    python experiments/tf_exclusion_demo.py                  # 3 seeds, ~25-45 min GPU (relay-safe)
    python experiments/tf_exclusion_demo.py --quick          # 1 seed, tiny grid (smoke)
    python experiments/tf_exclusion_demo.py --selftest       # pure-function unit tests

Pre-registration (2026-09-30, criteria before data; this docstring is the pre-registration record)
-----------------------------------------------------------------------------------------------
(1) Task and model: MNIST cut into 7x7 patches -> 16 tokens (the project's existing TinyGPT: causal
    multi-head attention, pre-LN residual blocks, d_model=48, n_head=3, n_layer=2, ~60k parameters).
    Reference (teacher) = TinyGPT with a LARGE pooled bottleneck (k_pool=256) so that H(Z_t) is
    measurable for the G0 codebook gate; students = the same model with the arm's bottleneck at K.
    Budget: MNIST subset 25% (15000 train) and 20 epochs for teacher and students alike (KD: CE + KL T=3).
(2) Arms (the ONLY difference is the bottleneck's location / the bypass) and per-arm grids (fixed BEFORE
    the run, sized to the question each arm answers):
      SER        k_pool=K, k_ffn=None, residual=True   -- the pipeline arm; full grid
                 K in [2,4,8,16,32,64,128,256] (8 levels: the certified exclusion, the wall, the looseness)
      FFN        k_ffn=K,  k_pool=None, residual=True  -- boundary arm; reduced grid K in [2,8,64]
                 (the boundary claim only needs K=2 plus two references)
      FFN-NORES  k_ffn=K,  k_pool=None, residual=False -- the T2 control; same reduced grid
    => 1 teacher + 8 + 3 + 3 = 15 units per seed, 45 units total (~75-80 min on the borrowed GPU).
(3) Readouts per (arm, K, seed): accuracy, gap g = acc_t - acc_s, relative gap rel = g/acc_t.
    Per seed the pipeline reports: R_IB = task_info_fano(1-acc_t, 10), K_min = ceil(e^R_IB),
    the included/excluded students from the K_min rule, K* (plateau + gap criterion), the measured wall
    K_wall (measured_wall, rel_tol=0.10) and looseness = ln(K_wall)/R_IB -- via preflight_report(..., r_ib=R_IB).
(4) Pre-registered criteria (pure functions; >=2/3 seeds for X1/X2):
    X1 (end-to-end exclusion works on the serial path): SER arm -- every K with log K < R_IB has
       rel-gap >= 0.25 (the certified exclusion is real, i.e. R1a holds), AND at least one K in the grid
       has rel-gap <= 0.10 (a wall exists inside the grid).
    X2 (the residual bypass unbinds a per-layer bottleneck): FFN arm at K=2 keeps rel-gap <= 0.10
       (accuracy retained even though log 2 = 0.69 << R_IB), WHILE FFN-NORES at K=2 has rel-gap >= 0.25 --
       same VQ, only the bypass differs, so the bypass (not the VQ) is the cause.
    X3 (wiring-table consequence, reported not tested): the SER arm's behaviour is consistent with
       min_capacity_serial([log K]) as the supply; the FFN arm's is NOT -- per the wiring table (P4,
       verified on MLPs) a bypassed branch is a parallel path, so the per-layer log K is not the supply
       at all. The quantitative statement is the measured pair (X1 vs X2).
    X4 (wall + looseness, new task family): report K_wall / looseness for the SER arm, adding
       "Transformer + MNIST" to the LOOSENESS survey (expect looseness ~1.2-1.6 at MNIST-level error).
    Verdict (pure function tf_verdict): SUPPORT if X1 and both halves of X2 hold; PARTIAL if exactly one
    of X1/X2 holds; else MIXED/REFUTED -- recorded as-is.
(5) Relay: per-unit checkpoints under experiments/data/tf/<fingerprint>/; --fresh recomputes.
(6) Iron rules: criteria before data; pure functions unit-testable; English-only output; no downloads
    (MNIST is in experiments/data/MNIST/raw).
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
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
from capacity_lens import (train_one_epoch, evaluate, entropy_of, task_info_fano,
                           DistillCapacity, preflight_report, measured_wall, looseness,
                           min_capacity_serial)
from mnist_demo import get_mnist
from transformer_demo import TinyGPT

N_CLASSES = 10
K_TEACHER = 256
K_GRID = [2, 4, 8, 16, 32, 64, 128, 256]      # SER (the pipeline arm)
K_GRID_FFN = [2, 8, 64]                        # FFN / FFN-NORES (the boundary arm)
DEFAULT_SUBSET = 0.25
DEFAULT_EPOCHS = 20
LR = 1e-3
SEED_TEACHER = 30000
SEED_STUDENT = 31000

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
JSONL = os.path.join(_DATA, "tf.jsonl")
LOG = os.path.join(_DATA, "tf.log")
REPORT = os.path.join(_DATA, "tf_report.txt")
CKPT_ROOT = os.path.join(_DATA, "tf")


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


def build_arm(arm, K):
    """SER: pooled serial bottleneck. FFN: FFN-branch bottleneck (bypassed by the residual stream).
    FFN-NORES: the same VQ with the bypass removed."""
    if arm == "ser":
        return TinyGPT(k_pool=K, k_ffn=None, residual=True)
    if arm == "ffn":
        return TinyGPT(k_pool=None, k_ffn=K, residual=True)
    if arm == "ffn_nores":
        return TinyGPT(k_pool=None, k_ffn=K, residual=False)
    raise ValueError(f"unknown arm {arm}")


# ---------------------------------------------------------------------------
# Criterion pure functions
# ---------------------------------------------------------------------------

def splits(acc_t, accs, r_ib):
    """Certified exclusion sets and measured sets for one seed (pure function).
    Below-K_min = excluded by the certified rule; rel gaps classify success/failure."""
    below = sorted(K for K in accs if math.log(K) < r_ib)
    rel = {K: (acc_t - accs[K]) / max(acc_t, 1e-12) for K in accs}
    fails = sorted(K for K in accs if rel[K] >= 0.25)
    succeeds = sorted(K for K in accs if rel[K] <= 0.10)
    return below, rel, fails, succeeds


def x1_holds(acc_t, accs, r_ib):
    """X1: certified exclusion is real on the serial path AND a wall is locatable."""
    below, rel, _fails, succeeds = splits(acc_t, accs, r_ib)
    excl_real = all(rel[K] >= 0.25 for K in below) and len(below) > 0
    return bool(excl_real and len(succeeds) > 0), below, rel


def x2_holds(ffn_acc_at2, ffn_nores_acc_at2, acc_t):
    """X2: bypassed bottleneck at K=2 keeps accuracy; the same VQ without the bypass collapses."""
    keep = (acc_t - ffn_acc_at2) / acc_t <= 0.10
    collapse = (acc_t - ffn_nores_acc_at2) / acc_t >= 0.25
    return bool(keep and collapse), keep, collapse


def tf_verdict(per_seed):
    """Pre-registered verdict (docstring (4)). per_seed: list of dicts with x1 / x2 flags."""
    n = len(per_seed)
    need = math.ceil(2 / 3 * n)
    n_x1 = sum(1 for r in per_seed if r["x1"])
    n_x2 = sum(1 for r in per_seed if r["x2"])
    detail = [f"  seeds: {n}; X1 (serial exclusion + wall) passed {n_x1}/{n}; "
              f"X2 (bypass unbinds / no-bypass binds) passed {n_x2}/{n}"]
    for r in per_seed:
        detail.append(f"    seed {r['seed']}: acc_t={r['acc_t']:.3f} R_IB={r['r_ib']:.2f} "
                      f"K_min={r['k_min']} K_wall={r['k_wall']} looseness={r['looseness']} "
                      f"| excluded={r['below']} fails={r['fails']} succeeds={r['succeeds']}")
        detail.append(f"      arm gaps (rel): SER={ {k: round(v, 2) for k, v in r['ser_rel'].items()} }")
        detail.append(f"                      FFN={ {k: round(v, 2) for k, v in r['ffn_rel'].items()} } "
                      f"FFN-NORES={ {k: round(v, 2) for k, v in r['nores_rel'].items()} }")
    if n_x1 >= need and n_x2 >= need:
        return ("SUPPORT: the exclusion pipeline runs end-to-end on a Transformer (certified exclusion + "
                "located wall on the pooled serial path), AND the Transformer-specific boundary is "
                "confirmed: a VQ inside the FFN branch is bypassed by the residual stream, so its log K is "
                "not the supply -- the wiring table's parallel rule governs there"), detail
    if n_x1 >= need or n_x2 >= need:
        return ("PARTIAL: one of the two Transformer claims holds (recorded as-is; see the per-seed lines)",
                detail)
    return "MIXED/REFUTED: neither claim held on this model/task (recorded as-is)", detail


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_signature(subset, epochs, grid, quick):
    raw = "|".join([f"sub{subset}", f"ep{epochs}", f"k{'/'.join(map(str, grid))}", f"q{int(quick)}",
                    "tinygpt-d48-h3-l2-p7"])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]


def train_unit(arm, K, seed, train_loader, test_loader, teacher, epochs, device, ckpt_path, resume):
    if resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        return ck["acc"], ck.get("h_z")
    torch.manual_seed(seed)
    model = build_arm(arm, K).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(epochs):
        train_one_epoch(model, train_loader, opt, device=device, distill_from=teacher)
    acc, idx = evaluate(model, test_loader, device=device)
    h_z = entropy_of(idx[0], K) if idx else None
    torch.save({"acc": acc, "h_z": h_z, "state_dict": model.state_dict()}, ckpt_path)
    return acc, h_z


def run_tf(seeds, subset, epochs, grid, resume, device):
    fp = run_signature(subset, epochs, grid, False)
    ckpt_dir = os.path.join(CKPT_ROOT, fp)
    os.makedirs(ckpt_dir, exist_ok=True)
    print("=" * 88)
    print(f"[TF-EXCLUSION] exclusion pipeline on a Transformer + the residual-bypass boundary")
    print("=" * 88)
    print(f"  [checkpoints] {'relay mode (cache hit -> skip)' if resume else 'recompute mode'}  {ckpt_dir}")
    print(f"  [model] TinyGPT (d48/h3/l2/p7, 16 tokens) ; subset={subset} epochs={epochs} ; K grid={grid}")
    print("  [arms] SER (pooled serial) / FFN (bypassed by the residual) / FFN-NORES (same VQ, no bypass)")
    train_loader, test_loader = get_mnist(subset=subset)
    per_seed = []
    for s in range(seeds):
        t0 = time.time()
        acc_t, h_z_t = train_unit("ser", K_TEACHER, SEED_TEACHER + s, train_loader, test_loader,
                                  None, epochs, device,
                                  os.path.join(ckpt_dir, f"teacher_s{s}.pt"), resume)
        teacher = build_arm("ser", K_TEACHER).to(device)
        teacher.load_state_dict(torch.load(os.path.join(ckpt_dir, f"teacher_s{s}.pt"),
                                          map_location=device, weights_only=True)["state_dict"])
        teacher.eval()
        r_ib = task_info_fano(1.0 - acc_t, N_CLASSES)
        k_min = DistillCapacity().min_codebook_size(r_ib)
        accs = {}
        for arm, arm_grid in (("ser", grid), ("ffn", K_GRID_FFN), ("ffn_nores", K_GRID_FFN)):
            accs[arm] = {}
            for K in arm_grid:
                acc, _ = train_unit(arm, K, SEED_STUDENT + 100 * s + K, train_loader, test_loader,
                                    teacher, epochs, device,
                                    os.path.join(ckpt_dir, f"{arm}_K{K}_s{s}.pt"), resume)
                accs[arm][K] = acc
        x1, below, ser_rel = x1_holds(acc_t, accs["ser"], r_ib)
        _, ffn_rel, _, _ = splits(acc_t, accs["ffn"], r_ib)
        _, nores_rel, _, _ = splits(acc_t, accs["ffn_nores"], r_ib)
        x2, keep2, collapse2 = x2_holds(accs["ffn"][grid[0]], accs["ffn_nores"][grid[0]], acc_t)
        _, _, fails, succeeds = splits(acc_t, accs["ser"], r_ib)
        k_wall, _rel = measured_wall(grid, accs["ser"], acc_t, rel_tol=0.10)
        lo = looseness(r_ib, k_wall)
        pf = preflight_report(acc_t, h_z_t or 0.0, K_TEACHER, grid, accs["ser"], acc_min=0.95, r_ib=r_ib)
        print(f"\n  seed {s}: teacher acc={acc_t:.3f} H(Z_t)={h_z_t:.2f} -> R_IB={r_ib:.2f} K_min={k_min} "
              f"[{time.time() - t0:.0f}s]")
        print(f"    SER   acc: " + "  ".join(f"K{K}:{accs['ser'][K]:.3f}" for K in grid))
        print(f"    FFN   acc: " + "  ".join(f"K{K}:{accs['ffn'][K]:.3f}" for K in K_GRID_FFN))
        print(f"    NR    acc: " + "  ".join(f"K{K}:{accs['ffn_nores'][K]:.3f}" for K in K_GRID_FFN))
        print(f"    certified exclusion (log K < R_IB) -> {below}; measured fails {fails}; succeeds {succeeds}")
        print(f"    X1 {'PASS' if x1 else 'FAIL'} (serial exclusion + wall) ; "
              f"X2 {'PASS' if x2 else 'FAIL'} (bypass keeps={keep2}, no-bypass collapses={collapse2})")
        print(f"    K_wall={k_wall} looseness={None if lo is None else round(lo, 2)} ; preflight: {pf['verdict']}")
        print(f"    supply reading: serial min_capacity_serial([log {grid[0]}]) = "
              f"{min_capacity_serial([math.log(grid[0])]):.2f} nats vs R_IB = {r_ib:.2f} nats ; "
              f"the FFN arm is bypassed by the residual stream, so the parallel rule (P4) governs there")
        jsonl_append(JSONL, dict(seed=s, arm_accs=accs, acc_t=acc_t, h_z_t=h_z_t, r_ib=r_ib, k_min=k_min,
                                 below=below, fails=fails, succeeds=succeeds, x1=x1, x2=x2,
                                 k_wall=k_wall, looseness=lo, verdict=pf["verdict"], fp=fp))
        per_seed.append(dict(seed=s, acc_t=acc_t, r_ib=r_ib, k_min=k_min, below=below, fails=fails,
                             succeeds=succeeds, x1=x1, x2=x2, k_wall=k_wall, looseness=lo,
                             ser_rel=ser_rel, ffn_rel=ffn_rel, nores_rel=nores_rel))
    print("\n  --- TF-EXCLUSION verdict ---")
    verdict, detail = tf_verdict(per_seed)
    for line in detail:
        print(line)
    print(f"  [verdict] {verdict}")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("[TF-EXCLUSION] Transformer exclusion pipeline + residual-bypass boundary\n")
        for line in detail:
            f.write(line + "\n")
        f.write("verdict: " + verdict + "\n")
    print(f"  [report] {REPORT}")
    return per_seed, verdict


def run_selftest():
    ok = []
    # splits: certified exclusion / measured sets
    acc_t = 0.96
    accs = {2: 0.12, 4: 0.30, 8: 0.90, 16: 0.955, 32: 0.958, 64: 0.959, 128: 0.960, 256: 0.960}
    r_ib = task_info_fano(1.0 - acc_t, N_CLASSES)
    below, rel, fails, succeeds = splits(acc_t, accs, r_ib)
    ok.append(("R_IB for acc 0.96 is ~2.05 nats", abs(r_ib - 2.05) < 0.05))
    ok.append(("certified exclusion covers K where log K < R_IB", below == [2, 4]))
    ok.append(("measured fails are the far-below points", fails == [2, 4]))
    ok.append(("measured success starts at/below the certified K_min", succeeds[0] == 8))
    ok.append(("X1 holds when exclusion is real and a wall exists", x1_holds(acc_t, accs, r_ib)[0]))
    bad = dict(accs)
    bad[4] = 0.94                     # a student below K_min that somehow matches the teacher
    ok.append(("X1 fails when an excluded student matches", not x1_holds(acc_t, bad, r_ib)[0]))
    ok.append(("X2 holds when the bypass keeps and no-bypass collapses",
               x2_holds(0.95, 0.20, acc_t)[0]))
    ok.append(("X2 fails when the bypass does not keep", not x2_holds(0.50, 0.20, acc_t)[0]))
    ok.append(("X2 fails when no-bypass does not collapse", not x2_holds(0.95, 0.90, acc_t)[0]))
    per = [dict(seed=0, x1=True, x2=True, acc_t=0.96, r_ib=r_ib, k_min=8, k_wall=16, looseness=1.4,
                below=[2, 4], fails=[2, 4], succeeds=[16, 32], ser_rel={}, ffn_rel={}, nores_rel={}),
           dict(seed=1, x1=True, x2=True, acc_t=0.96, r_ib=r_ib, k_min=8, k_wall=16, looseness=1.4,
                below=[2, 4], fails=[2, 4], succeeds=[16, 32], ser_rel={}, ffn_rel={}, nores_rel={}),
           dict(seed=2, x1=True, x2=True, acc_t=0.96, r_ib=r_ib, k_min=8, k_wall=16, looseness=1.4,
                below=[2, 4], fails=[2, 4], succeeds=[16, 32], ser_rel={}, ffn_rel={}, nores_rel={})]
    v, _ = tf_verdict(per)
    ok.append(("verdict SUPPORT when both claims hold 3/3", v.startswith("SUPPORT")))
    per[1]["x2"] = False
    per[2]["x2"] = False          # only one claim reaches 2/3
    v, _ = tf_verdict(per)
    ok.append(("verdict PARTIAL when only one claim reaches 2/3", v.startswith("PARTIAL")))
    per[1]["x1"] = False
    per[2]["x1"] = False          # neither claim reaches 2/3
    v, _ = tf_verdict(per)
    ok.append(("verdict MIXED when neither reaches 2/3", v.startswith("MIXED")))
    print("[TF selftest] pure-function unit tests:")
    all_ok = True
    for name, passed in ok:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        all_ok = all_ok and passed
    print(f"  [selftest verdict] {'all PASS' if all_ok else 'FAILURES present (code defect)'}")
    return all_ok


def main():
    ap = argparse.ArgumentParser(description="TF-EXCLUSION: Transformer exclusion pipeline + bypass boundary")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--subset", type=float, default=DEFAULT_SUBSET)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--quick", action="store_true", help="1 seed, tiny grid, few epochs (smoke)")
    ap.add_argument("--fresh", action="store_true", help="recompute (default: relay mode)")
    ap.add_argument("--selftest", action="store_true", help="pure-function unit tests (no training)")
    args = ap.parse_args()
    if args.selftest:
        run_selftest()
        return
    _setup_logging()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if args.quick:
        run_tf(1, 0.1, 4, [2, 8, 256], not args.fresh, device)
    else:
        run_tf(args.seeds, args.subset, args.epochs, K_GRID, not args.fresh, device)
    print("\nDone.")


if __name__ == "__main__":
    main()
