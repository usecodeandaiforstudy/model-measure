"""PA1: the P-A1-3 budget-ladder judgment (K_t=128 effective-capacity audit, grid64 CNN-euclid).

Answers the highest-priority open item of docs/framework.md section 8: A-001's corollary says
"if output-space distinguishability is a function of capacity, then 'capacity sufficient but cannot spread out'
cases should not exist". The v3.0 K_t=128 layer is the suspected counterexample: nominal capacity
ln 128 ~= 4.85 nats > task H(Y) = ln 64 ~= 4.16 nats, yet after the 80ep recalibration acc is still 0.83-0.90,
failing the G0 acc gate. This script extends the archived 40/80ep curve with 120/160ep (and beyond) rungs
and applies the pre-registered decision rule below.

Usage:
    python experiments/pa1_demo.py                     # ladder to 160ep, 3 seeds, rungs every 40ep
    python experiments/pa1_demo.py --max-epochs 200    # extend the ladder (continues from cached rungs)
    python experiments/pa1_demo.py --fresh             # recompute from epoch 1 (default: relay mode -- each
                                                       # seed continues from its highest cached rung)
    python experiments/pa1_demo.py --selftest          # pure-function unit tests (no training)

Data: synthetic grid64 image task generated in-process (make_grid64_images, n_per_class=150, data seed = s);
no dataset files, no downloads. On a GPU one seed x 160ep is a few minutes.

Pre-registration (2026-09-29, criteria before data; this docstring is the pre-registration record)
-----------------------------------------------------------------------------------------------
(1) Motivation: framework.md P-A1-3. The effective-capacity audit on the archived data is already done:
    40ep H(Z_t) = 4.218/4.150/4.248 (acc 0.829/0.725/0.816); 80ep H(Z_t) = 4.32/4.29/4.25
    (acc 0.903/0.827/0.867). Both rungs sit ABOVE the task wall ln64=4.159 and above the G0-acc requirement
    Fano(0.95)=3.75 nats -> prima facie "capacity sufficient" -> the counterexample candidate is live.
    But the 40->80ep acc gain (+0.05..+0.10) is positive: "slow reachability" (axis 3, A-002) cannot be
    excluded from two points. Two points cannot separate a slow climb from a hard plateau; a ladder can.

(2) Design: bit-faithful unit construction to cp3_multilevel.py (the archived curve family):
    - task: make_data("cnn", N_PER_CLASS=150, s) from cp3_demo (make_grid64_images, data seed = s);
    - model: build_model("cnn", K_T=128, "euclid") = GridCNN(d=64, n_codes=128, metric="euclid");
    - init seed: torch.manual_seed(14000 + 100*CELL_IDX + s) with CELL_IDX=2 (CP3's CNN-euclid formula);
    - optimizer: Adam lr=1e-3 (CP3 CNN convention); batch 128 train / 256 test; same train_one_epoch.
    ONE deviation from the project's "one unit = one training run" convention, and the reason: the ladder
    is a TRAJECTORY question (does acc plateau or keep climbing), so each seed is ONE trajectory with a
    rung checkpoint every --step epochs. The plateau discriminator (delta between adjacent rungs of the
    SAME trajectory) is noise-free, whereas delta between independent from-scratch units would carry the
    +-0.03 run-to-run noise and could not separate a +0.03/40ep climb from a flat plateau. A resume
    continuation is a valid trajectory (batch order stays deterministic from the seeded RNG), and the
    verdict only reads trajectory shape. The archived 40/80 rows remain the independent-unit reference.
(3) Readouts per rung: acc, H(Z_t) (entropy_of), utilization, R_IB (Fano from the rung's acc), G0 tag
    (acc>=0.95 AND H(Z_t)>=0.8*ln128), wall distance H(Z_t)-ln64, achieved-I gap H(Z_t)-R_IB.
(4) Pre-registered decision rule (pure function pa1_verdict; rungs = per-seed rows of ONE trajectory):
    per seed, last rung L and the rung before it give acc_L and delta = acc_L - acc_prev (delta is
    undefined for a single-rung seed -> that seed counts for neither side):
      Case A (slow reachability, supports A-001, no counterexample):
        >= 2/3 seeds with (acc_L >= 0.95) OR (delta >= 0.03)   -- still climbing toward the gate
      Case B (hard plateau: capacity sufficient but cannot spread; A-001's corollary needs a qualifier):
        >= 2/3 seeds with (acc_L < 0.95) AND (delta < 0.01) AND (H(Z_t)_L >= Fano(0.95)=3.75)
      Otherwise: gray zone -> recorded honestly, recommended action = extend the ladder (--max-epochs).
    Descriptive columns only (no verdict rides on them): wall distance, achieved-I gap.
(5) Archived reference rows (from cp3_multilevel.jsonl / report, printed for continuity, never re-judged):
    40ep acc 0.829/0.725/0.816 (H 4.22/4.15/4.25); 80ep acc 0.903/0.827/0.867 (H 4.32/4.29/4.25).
    The fresh rungs are measured on this machine/device and may differ by run-to-run noise (+-0.03, see
    cph_demo docstring (8)) -- the comparison is descriptive.
(6) Seed scheme: teacher init = 14200 + s, data seed = s (identical to cp3_multilevel's archived units).
    Non-overlapping with the other families.
(7) Iron rules: print only GBK-encodable characters; criteria before data (this docstring); pure functions
    unit-testable (--selftest); negative results archived honestly; no downloads, no dataset files.
(8) Relay / cache (default mode since 2026-09-29): rung checkpoints live in
    experiments/data/pa1/<fingerprint>/kt128_teacher_s<seed>_ep<e>.pt with {state_dict, acc, h_z, epochs}.
    Each run continues every seed from its highest cached rung <= --max-epochs (a 160ep run later extended to
    200ep starts at 160), and cached rungs are re-read from the JSONL so a fully-cached re-run still prints
    the ladder and judges it. --fresh ignores all caches and recomputes from epoch 1 (use once after changing
    the training code). The fingerprint contains (K_t, n_per_class, lr, device) and deliberately NOT
    --max-epochs/--step: extending the ladder must reuse earlier rungs. Keep --step fixed within a campaign.
(9) Cost: synthetic task, ~60 batches/epoch; on the borrowed GPU one seed x 160ep is a few minutes,
    3 seeds well under 20 minutes. CPU is also fine (40ep ~= 2 min per the archive).
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
from capacity_lens import train_one_epoch, evaluate, entropy_of, task_info_fano
from mnist_demo import g0_gate
from cp3_demo import make_data, build_model

K_T = 128
C_TASK = 64
N_PER_CLASS = 150
CELL_IDX = 2
DEFAULT_MAX = 160
DEFAULT_STEP = 40
LN64 = math.log(C_TASK)
FANO_095 = task_info_fano(0.05, C_TASK)   # Fano requirement for acc=0.95 on grid64 (~3.75 nats)
ARCHIVED = {
    40: [(0.829, 4.218), (0.725, 4.150), (0.816, 4.248)],
    80: [(0.903, 4.32), (0.827, 4.29), (0.867, 4.25)],
}

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
JSONL = os.path.join(_DATA, "pa1.jsonl")
LOG = os.path.join(_DATA, "pa1.log")
REPORT = os.path.join(_DATA, "pa1_report.txt")
CKPT_ROOT = os.path.join(_DATA, "pa1")


class _Tee:
    """stdout/stderr to both the terminal and the log file (host capture is unreliable)."""

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


def jsonl_append(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def run_signature(lr, device):
    """Config fingerprint: K_t / n_per_class / lr / device. Deliberately excludes --max-epochs and --step
    (ladder extension must reuse earlier rungs; see docstring (8))."""
    dev = "cuda" if str(device).startswith("cuda") else "cpu"
    raw = "|".join([f"kt{K_T}", f"npc{N_PER_CLASS}", f"lr{lr}", f"dev{dev}"])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]


def ckpt_dir_for(lr, device):
    d = os.path.join(CKPT_ROOT, run_signature(lr, device))
    os.makedirs(d, exist_ok=True)
    return d


def rung_path(ckpt_dir, s, e):
    return os.path.join(ckpt_dir, f"kt{K_T}_teacher_s{s}_ep{e}.pt")


# ---------------------------------------------------------------------------
# Criterion pure function (independently unit-testable)
# ---------------------------------------------------------------------------

def pa1_verdict(records):
    """The P-A1-3 decision rule (pure function, see docstring (4)).

    records: list of dicts with keys s, epochs, acc, h_z (one row per (seed, rung) of the same trajectory).
    Returns (verdict_string, detail_lines)."""
    by_seed = {}
    for r in records:
        by_seed.setdefault(r["s"], []).append(r)
    if not by_seed:
        return "UNDECIDABLE: no ladder data", ["  no records"]
    need = math.ceil(2 / 3 * len(by_seed))
    na, nb, single = 0, 0, 0
    detail = []
    for s in sorted(by_seed):
        rows = sorted(by_seed[s], key=lambda r: r["epochs"])
        last = rows[-1]
        prev = rows[-2] if len(rows) >= 2 else None
        if prev is None:
            single += 1
            detail.append(f"  seed {s}: single rung only (ep{last['epochs']}), delta undefined, counts for neither side; "
                          f"acc={last['acc']:.3f} H={last['h_z']:.2f}")
            continue
        delta = last["acc"] - prev["acc"]
        a = last["acc"] >= 0.95 or delta >= 0.03
        b = (last["acc"] < 0.95 and delta < 0.01 and last["h_z"] >= FANO_095)
        na += 1 if a else 0
        nb += 1 if b else 0
        detail.append(f"  seed {s}: ep{prev['epochs']}->ep{last['epochs']} "
                      f"acc {prev['acc']:.3f}->{last['acc']:.3f} (delta={delta:+.3f})  "
                      f"H={last['h_z']:.2f} -> {'A side' if a else ('B side' if b else 'neither side')}")
    if na >= need:
        return ("A: slow reachability (acc still climbing toward the 0.95 gate) -- SUPPORT A-001, K_t=128 is not a counterexample; "
                "it is the axis-3 'thin margin + high cost' case", detail)
    if nb >= need:
        return ("B: capacity sufficient but cannot spread out (acc hard plateau < 0.95 and H(Z_t) >= Fano(0.95)) -- "
                "A-001's corollary needs a qualifier (distinguishability = function of capacity and function of reachability)", detail)
    return "MIXED: not enough to conclude (recommend raising --max-epochs to extend the ladder)", detail


# ---------------------------------------------------------------------------
# Ladder training
# ---------------------------------------------------------------------------

def train_seed_ladder(s, max_epochs, step, lr, device, ckpt_dir, resume):
    """One seed = one trajectory with a rung every `step` epochs. Returns the list of fresh rung rows.
    Default (relay) mode continues from the highest cached rung <= max_epochs (documented deviation from the
    one-unit convention: the ladder is a trajectory question, see docstring (2)/(8)); --fresh recomputes
    from epoch 1."""
    torch.manual_seed(14000 + 100 * CELL_IDX + s)
    (xtr, ytr), (xte, yte) = make_data("cnn", N_PER_CLASS, s)
    train_loader = DataLoader(list(zip(xtr, ytr)), batch_size=128, shuffle=True)
    test_loader = DataLoader(list(zip(xte, yte)), batch_size=256)
    model = build_model("cnn", K_T, "euclid").to(device)
    start = 0
    if resume:
        for e in range(max_epochs - max_epochs % step, 0, -step):
            p = rung_path(ckpt_dir, s, e)
            if e <= max_epochs and os.path.exists(p):
                ck = torch.load(p, map_location=device, weights_only=True)
                model.load_state_dict(ck["state_dict"])
                start = e
                print(f"  seed {s}: [resumed from ep{start}] acc={ck['acc']:.3f}  H(Z_t)={ck['h_z']:.2f}")
                break
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rows = []
    t0 = time.time()
    for ep in range(start + 1, max_epochs + 1):
        train_one_epoch(model, train_loader, opt, device=device)
        if ep % step == 0:
            acc, idx = evaluate(model, test_loader, device=device)
            h_z = entropy_of(idx[0], K_T)
            util = model.vq.codebook_utilization(idx[0])
            r_ib = task_info_fano(1.0 - acc, C_TASK)
            g0, ok_acc, ok_cb = g0_gate(acc, h_z, K_T)
            torch.save({"state_dict": model.state_dict(), "acc": acc, "h_z": h_z,
                        "epochs": ep}, rung_path(ckpt_dir, s, ep))
            row = dict(s=s, k_t=K_T, epochs=ep, acc=acc, h_z=h_z, util=util,
                       g0=g0, r_ib=r_ib, wall_dist=h_z - LN64, gap=h_z - r_ib,
                       fp=run_signature(lr, device))
            rows.append(row)
            jsonl_append(JSONL, row)
            g0_tag = "PASS" if g0 else f"FAIL acc{'PASS' if ok_acc else 'FAIL'}/codebook{'PASS' if ok_cb else 'FAIL'}"
            print(f"  seed {s} ep{ep}: acc={acc:.4f}  H(Z_t)={h_z:.2f}  util={util:.0%}  "
                  f"G0 {g0_tag}  R_IB={r_ib:.2f}  wall_dist={h_z - LN64:+.2f}  gap={h_z - r_ib:.2f}"
                  f"  [{time.time() - t0:.0f}s]")
    return rows


def run_pa1(max_epochs, step, seeds, lr, resume, device):
    ckpt_dir = ckpt_dir_for(lr, device)
    fp = run_signature(lr, device)
    print("\n" + "=" * 76)
    print(f"[PA1] P-A1-3 budget ladder: K_t=128 effective-capacity audit (grid64 CNN-euclid; {seeds} seeds; "
          f"ladder {step}ep -> {max_epochs}ep; lr={lr})")
    print("=" * 76)
    print(f"  [checkpoint] {'relay mode (auto resume/skip)' if resume else 'recompute mode (--fresh: recompute from ep1)'}  {ckpt_dir}")
    print(f"  [background] nominal capacity ln128={math.log(K_T):.2f} > task wall ln64={LN64:.2f} nats;")
    print(f"        80ep archived acc 0.83-0.90 fails the gate, effective capacity 4.25-4.32 > Fano(0.95)={FANO_095:.2f}.")
    print("  [criterion] A=slow reachability (SUPPORT A-001) / B=capacity sufficient but cannot spread out (counterexample) / MIXED (see docstring (4))")
    # cached rungs of this config (relay mode: a fully-cached re-run still prints and judges the ladder)
    cached = {}
    if os.path.exists(JSONL):
        with open(JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if (r.get("fp") == fp and r.get("epochs", 1 << 30) <= max_epochs
                        and r["epochs"] % step == 0):
                    cached[(r["s"], r["epochs"])] = r
    rows = []
    for s in range(seeds):
        print(f"  --- s{s} ---")
        rows.extend(train_seed_ladder(s, max_epochs, step, lr, device, ckpt_dir, resume))
    seen = {(r["s"], r["epochs"]) for r in rows}
    for k, r in sorted(cached.items()):
        if k not in seen:
            rows.append(r)
            print(f"  [cached rung] s{r['s']} ep{r['epochs']}: acc={r['acc']:.4f}  "
                  f"H(Z_t)={r['h_z']:.2f}")
    rows.sort(key=lambda r: (r["s"], r["epochs"]))
    print("\n  --- archived reference (40/80ep, independent units; differences include run noise, descriptive only) ---")
    for e, triples in sorted(ARCHIVED.items()):
        print(f"  {e}ep: " + "  ".join(f"acc={a:.3f}/H={h:.2f}" for a, h in triples))
    print(f"  reference lines: wall ln64={LN64:.2f} nats; Fano(0.95)={FANO_095:.2f} nats; "
          f"codebook gate 0.8*ln128={0.8 * math.log(K_T):.2f}")
    verdict, detail = pa1_verdict(rows)
    print("\n  --- P-A1-3 verdict ---")
    for line in detail:
        print(line)
    print(f"  [verdict] {verdict}")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("[PA1] P-A1-3 budget ladder verdict report\n")
        f.write(f"config: K_t={K_T} seeds={seeds} ladder={step}->{max_epochs} lr={lr} dev={device}\n\n")
        for row in rows:
            f.write(f"s{row['s']} ep{row['epochs']}: acc={row['acc']:.4f} H={row['h_z']:.3f} "
                    f"R_IB={row['r_ib']:.3f} wall_dist={row['wall_dist']:+.3f} gap={row['gap']:+.3f} "
                    f"g0={row['g0']}\n")
        f.write("\nverdict: " + verdict + "\n")
    print(f"  [report] {REPORT}")
    return rows, verdict


def run_selftest():
    """Pure-function unit tests (no training, no data)."""
    ok = []
    ok.append(("FANO_095 ~ 3.75 nats", abs(FANO_095 - 3.753) < 0.01))
    # Case A: still climbing (delta >= 0.03)
    rows_a = []
    for s in range(3):
        rows_a.append(dict(s=s, epochs=120, acc=0.84, h_z=4.3))
        rows_a.append(dict(s=s, epochs=160, acc=0.875, h_z=4.35))
    v, _ = pa1_verdict(rows_a)
    ok.append(("case A climbing -> side A", v.startswith("A: ")))
    # Case B: hard plateau, capacity sufficient
    rows_b = []
    for s in range(3):
        rows_b.append(dict(s=s, epochs=120, acc=0.865, h_z=4.3))
        rows_b.append(dict(s=s, epochs=160, acc=0.869, h_z=4.35))
    v, _ = pa1_verdict(rows_b)
    ok.append(("case B plateau -> side B", v.startswith("B: ")))
    # Gray: mixed (1 climbing, 2 in-between: delta in (0.01, 0.03) counts for neither side)
    rows_g = [dict(s=0, epochs=120, acc=0.84, h_z=4.3), dict(s=0, epochs=160, acc=0.875, h_z=4.35),
              dict(s=1, epochs=120, acc=0.86, h_z=4.3), dict(s=1, epochs=160, acc=0.875, h_z=4.35),
              dict(s=2, epochs=120, acc=0.86, h_z=4.3), dict(s=2, epochs=160, acc=0.875, h_z=4.35)]
    v, _ = pa1_verdict(rows_g)
    ok.append(("mixed -> MIXED", v.startswith("MIXED")))
    # single-rung seed does not count
    rows_s = [dict(s=0, epochs=160, acc=0.86, h_z=4.3),
              dict(s=1, epochs=160, acc=0.87, h_z=4.3),
              dict(s=2, epochs=120, acc=0.86, h_z=4.3), dict(s=2, epochs=160, acc=0.865, h_z=4.35)]
    v, _ = pa1_verdict(rows_s)
    ok.append(("single-rung seed counts for neither side", v.startswith("MIXED") or v.startswith("UNDECIDABLE")))
    print("[PA1 selftest] pure-function unit tests:")
    all_ok = True
    for name, passed in ok:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        all_ok = all_ok and passed
    print(f"  [selftest verdict] {'all PASS' if all_ok else 'some FAIL (code defect)'}")
    return all_ok


def main():
    ap = argparse.ArgumentParser(description="PA1: P-A1-3 budget ladder (K_t=128 effective-capacity audit)")
    ap.add_argument("--max-epochs", type=int, default=DEFAULT_MAX,
                    help=f"ladder top (default {DEFAULT_MAX}; to extend, raise it and rerun the same command -- relay mode resumes automatically)")
    ap.add_argument("--step", type=int, default=DEFAULT_STEP,
                    help=f"rung interval (default {DEFAULT_STEP}; keep fixed within a campaign)")
    ap.add_argument("--seeds", type=int, default=3, help="number of seeds (default 3)")
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate (CNN convention 1e-3)")
    ap.add_argument("--fresh", action="store_true",
                    help="recompute mode: retrain from ep1 (default relay mode: resume each seed from its highest cached rung)")
    ap.add_argument("--selftest", action="store_true", help="pure-function unit tests (no training)")
    args = ap.parse_args()
    if args.selftest:
        run_selftest()
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    run_pa1(args.max_epochs, args.step, args.seeds, args.lr, not args.fresh, device)
    print("\nDone.")


if __name__ == "__main__":
    main()
