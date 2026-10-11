"""CP3: metric intervention experiment (2x2 factor: architecture x VQ metric), dissecting the "architecture family" composite variable.

Usage:
    python experiments/cp3_demo.py --quick         # smoke test (no verdict)
    python experiments/cp3_demo.py --fast          # reduced data tier but with verdict
    python experiments/cp3_demo.py                 # full tier (150/class) with verdict
    python experiments/cp3_demo.py --resume        # resume training: skip already-completed units

Resume training (see section (9)): the teacher of each (cell, seed) and the student of each (cell, seed, K)
is checkpointed as soon as it finishes training (experiments/data/cp3/<config fingerprint>/).
After an interruption, rerunning with --resume continues from the checkpoint; without --resume everything is
recomputed and the cache overwritten.

Pre-registration (2026-09-21, criteria before data; this docstring is the pre-registration record)
--------------------------------------------------------------------
(1) Motivation: CP1 left the "architecture family" puzzle -- arm1 (Classifier3D, euclid) locatable,
    arm2 (GridCNN, cosine) not_locatable, arm3 (GridMLP, euclid) locatable.
    CP1 (silhouette) and CP2 (codebook margin) twice failed to predict at the **representation statistic** level
    (CP1's prediction was falsified; CP2 was falsified and the full rerun agreed). Both point to:
    locatability is not a function of representation statistics. CP3 instead dissects the composite variable "architecture family"
    itself, decomposing it into two factors: architecture (MLP / CNN) x VQ metric (euclid / cosine).

(2) Design (2x2 factor, with the teacher/student metric consistent within a run -- isomorphic to CP1):
        Baseline cells (known): (MLP, euclid) = CP1 arm3 / CP2 all doses, known locatable
                      (CNN, cosine) = CP1 arm2,                 known not_locatable
        New cells (intervention):   (MLP, cosine)   <- the only variable relative to the baseline cell = metric
                      (CNN, euclid)   <- the only variable relative to the baseline cell = metric
    Each new cell is an **architecture-internal single-variable (metric) intervention**, with everything else (task, depth, BN,
    parameter count, optimizer, budget, seed scheme) bit-for-bit unchanged.
    Cross-architecture claims forbidden: MLP uses 3-dimensional grid64 and CNN uses the 16x16 image version of grid64,
    so MLP<->CNN simultaneously changes data modality/depth/BN/parameter count, not a single variable, and no architecture conclusion is drawn.

(3) Readouts (per cell per seed): teacher acc (G0 gate >=0.95), H(Z_t) (entropy_of,
    same convention as CP1/CP2), R_IB(Fano) -> K_min; student gap curve
    g(K) = acc_t - acc_s(K) (the K grid is anchored at each cell's seed 0 K_min, structurally homologous
    to CP2), and the criteria distill_plateau_ok / distill_gap_k_star / locatability
    are fully reused from CP1/CP2 (pure functions, unchanged).

(4) Pre-registered predictions (before data): two competing hypotheses, which the two new cells directly discriminate.
    H_metric (main hypothesis): the metric determines locatability. cosine discards magnitude information -- class clusters that are
      close in direction but different in scale are judged the same codeword (non-local identification), and the sharp covering
      wall of log K is flattened -> the cliff is not locatable.
        Prediction: (MLP, cosine) = not_locatable; (CNN, euclid) = locatable.
    H_architecture (competing hypothesis): architecture determines locatability, the metric is irrelevant.
        Prediction: (MLP, cosine) = locatable; (CNN, euclid) = not_locatable.
    Other results are mixed and archived honestly:
      - both new cells not_locatable -> the metric and the architecture are both independent obstacles (each sufficient to kill);
      - both new cells locatable    -> locatability is determined by a deeper factor (converging with CP2);
      - not falling into any of the above patterns -> recorded as "does not fall into a pre-registered pattern".
    The baseline cells also serve as a **reproduction check**: if (MLP, euclid) is not locatable or
    (CNN, cosine) is not not_locatable, report a reproduction failure rather than force an explanation.

(5) G0 gate and invalidity handling: any cell seed whose teacher fails the G0 double gate (acc < 0.95 or
    H(Z_t) < 0.8*ln(K_t), see g0_gate in mnist_demo/capacity_lens.preflight) -> that cell is invalid (archived honestly,
    the prediction cannot be tested). The acc threshold stays 0.95 (consistent with CP1/CP2, not adjusted to the result);
    the codebook gate was added 2026-09-22 (after the CP3 archived run): the C3 v3 audit found that the old single gate
    let through a collapsed teacher with acc=0.961 hugging the threshold but H(Z_t)=4.18=ln64 -- a teacher whose codebook has collapsed
    onto the wall is definitionally useless for measuring "whether there is a cliff below the wall". This experiment has archived 12 teachers
    (H(Z_t) 5.18-5.39) all passing the double gate, so the archived conclusion is unchanged; the gate only constrains future runs.
    Identified risk (from a calibration probe, before any verdict data): cosine acts on the
    relu non-negative features before VQ (the positive orthant, where cosine similarity is bounded in [0,1]), so capacity is limited -- MLP+cosine
    at 100/class gave 0.971/0.955/0.949 over three seeds (hugging the threshold with seed jitter). Accordingly, the verdict
    run uses a 150/class data budget (more data stabilizes its upper bound better than more epochs), expecting all three seeds
    to pass the gate; if a seed still fails the gate, that cell is honestly recorded as invalid, with no retroactive adjustment of the threshold.

(6) Seed scheme: teacher = 14000 + 100*cell index + s; student = 15000 +
    100*cell index + 100*s + K; data seed = s (the task's nominal information unchanged).
    Non-overlapping with the parent experiment (the 4000/5000/6000/7000 family), CP1 (the 9000/9500 family),
    and CP2 (the 12000/13000 family).

(7) Iron rules: print uses only GBK-encodable characters; criteria before data (this docstring is the pre-registration);
    pure functions independently unit-testable; negative results archived honestly; no cross-architecture claims.

(8) Budget calibration record (2026-09-21, before any verdict data was produced; the calibration target is the G0 gate's
    input = teacher convergence, never touching the locatability readouts):
    Two rounds of convergence probes (100/class, seed 0-2; fast data tier) results --
        mlp euclid  0.977@20  0.982@40  0.984@60             (converges easily)
        mlp cosine  seed0 0.959@80 0.971@120; seed1 0.955@120; seed2 0.949@120
        cnn euclid  0.595@3  0.637@6  0.859@10  0.909@15  0.907@25  0.999@40
        cnn cosine  0.946@3  0.983@10  1.000@15
    Accordingly, **before the verdict data was produced**, three items were changed (the prediction table (4) and the verdict rules were not modified at all):
      a) Epoch budget: CP1 arm spec (MLP 40 / CNN 6) -> calibrated values (MLP 120 / CNN 40).
         The original budget would leave CNN-euclid badly under-trained (0.637) and MLP-cosine unconverged,
         making both key cells invalid for reasons unrelated to the metric. Both metrics of the same architecture are applied equally.
      b) K grid: "adaptive with K_min" -> the fixed grid CALIBRATED_K shared by all four cells,
         eliminating the extra variable of grid differences.
      c) Data budget: 100/class -> 150/class, targeting the cosine threshold-hugging risk of (5)
         (more data stabilizes the accuracy upper bound better than more epochs).

(9) Resume training: a verdict run takes hours, so each unit is checkpointed as soon as it finishes training:
        experiments/data/cp3/<config fingerprint>/<cell>_teacher_s<seed>.pt   (state_dict + acc + h_z)
        experiments/data/cp3/<cell>_student_s<seed>_K<K>.pt        (acc)
    The config fingerprint is generated by run_signature (number of seeds, data tier, the two architectures' epochs, the K grid,
    teacher codebook size), and **different configs do not contaminate each other** -- otherwise a quick smoke test's cache would be
    misused by a full run. Each unit sets its seed at the start of the function and does not depend on global RNG state,
    so "resumed run" and "continuous run" are unit-by-unit equivalent (teacher weights are replayed via state_dict,
    and the distillation target is exactly the same). Granularity: an interruption loses at most one unit (the student of one K).
    Usage: after an interruption `python experiments/cp3_demo.py --resume`.
    This mechanism only affects computation scheduling and does not change any criterion, prediction, or readout definition.
--------------------------------------------------------------------
"""

import argparse
import hashlib
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
from capacity_lens import (VQBottleneck, train_one_epoch, evaluate, entropy_of,
                           task_info_fano)
from mnist_demo import (make_grid64, distill_gap_k_star, distill_plateau_ok,
                        g0_gate)
from cifar_demo import GridCNN, make_grid64_images


# ---------------------------------------------------------------------------
# Shared helpers (moved here from cp1_demo 2026-09-30; cp3_demo is the shared
# model/data dependency of pa1/pa3/pa4/moe/opt, cp1_demo was removed as a
# falsified experiment)
# ---------------------------------------------------------------------------

class GridMLP(nn.Module):
    """Linear(3->64) -> ReLU -> VQ(64, K, metric) -> Linear(64->64).

    metric defaults to "euclid"; the CP3 metric intervention uses "cosine" (the only variable = metric).
    """

    def __init__(self, n_codes=256, metric="euclid"):
        super().__init__()
        self.fc1 = nn.Linear(3, 64)
        self.vq = VQBottleneck(64, n_codes, metric=metric)
        self.fc2 = nn.Linear(64, 64)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        z, idx = self.vq(h)
        return self.fc2(z), [idx]


def locatability(records):
    """Arm-level locatability (pure function).

    records: [(plateau_ok, k_star), ...] one per seed; locatable := the fraction of seeds satisfying
    "plateau_ok and K* not None" >= 2/3.
    """
    ok = sum(1 for p, k in records if p and k is not None)
    return "locatable" if ok / max(len(records), 1) >= 2 / 3 else "not_locatable"


CELLS = [
    dict(name="MLP-euclid", arch="mlp", metric="euclid"),
    dict(name="MLP-cosine", arch="mlp", metric="cosine"),
    dict(name="CNN-euclid", arch="cnn", metric="euclid"),
    dict(name="CNN-cosine", arch="cnn", metric="cosine"),
]
KEY_CELLS = ["MLP-cosine", "CNN-euclid"]   # the two new cells, directly discriminating the two hypotheses
TEACHER_CODES = 256


def cell_budget(arch, quick):
    """Training budget per architecture (pure function, independently unit-testable).

    quick = smoke-test budget (convergence not sought); non-quick uses the **calibrated budget** (see the module docstring's
    "budget calibration record"): MLP 120 ep, CNN 40 ep. The calibrated budget applies the two metrics of the same architecture
    **equally**, thereby preserving the experimental purity of "the only variable = metric".
    """
    if arch == "mlp":
        return dict(epochs=25 if quick else 120, lr=1e-2)
    return dict(epochs=3 if quick else 40, lr=1e-3)


def make_data(arch, n_per_class, seed):
    """Fetch the task data by architecture (MLP=3-dimensional grid64; CNN=the 16x16 image version of grid64)."""
    if arch == "mlp":
        return make_grid64(n_per_class=n_per_class, seed=seed)
    return make_grid64_images(n_per_class=n_per_class, seed=seed)


def build_model(arch, n_codes, metric):
    """Build the model by architecture (the only degrees of freedom = K and the metric)."""
    if arch == "mlp":
        return GridMLP(n_codes=n_codes, metric=metric)
    return GridCNN(d=64, n_codes=n_codes, metric=metric)


# ---------------------------------------------------------------------------
# Criterion pure functions (independently unit-testable)
# ---------------------------------------------------------------------------

def metric_prediction():
    """The pre-registered prediction of H_metric (pure function, independently unit-testable)."""
    return {"MLP-cosine": "not_locatable", "CNN-euclid": "locatable"}


def arch_prediction():
    """The pre-registered prediction of H_architecture (pure function, independently unit-testable)."""
    return {"MLP-cosine": "locatable", "CNN-euclid": "not_locatable"}


def hypothesis_verdict(actual):
    """Decide the hypothesis from the key-cell measurements (pure function, independently unit-testable).

    actual: {cell name: "locatable"/"not_locatable"}; missing or None = that cell is invalid (G0 not passed).
    Returns (verdict string, list of detail lines).
    """
    vals = {k: actual.get(k) for k in KEY_CELLS}
    detail = [f"  {k}: {vals[k]}" for k in KEY_CELLS]
    if any(v is None for v in vals.values()):
        return "UNDECIDABLE (key cells invalid, G0 not passed)", detail
    mp, ap = metric_prediction(), arch_prediction()
    if all(vals[k] == mp[k] for k in KEY_CELLS):
        return "SUPPORT H_metric (metric determines locatability)", detail
    if all(vals[k] == ap[k] for k in KEY_CELLS):
        return "SUPPORT H_architecture (architecture determines locatability)", detail
    if all(v == "not_locatable" for v in vals.values()):
        return "MIXED: both new cells not_locatable (metric and architecture are both independent obstacles)", detail
    if all(v == "locatable" for v in vals.values()):
        return "MIXED: both new cells locatable (locatability is determined by a deeper factor)", detail
    return "MIXED: result does not fall into any pre-registered pattern (archived honestly)", detail


# ---------------------------------------------------------------------------
# Phase A / B
# ---------------------------------------------------------------------------

def train_teacher(cell, c_idx, s, epochs, n_per_class, lr, device="cpu",
                  ckpt_dir=None, resume=False):
    """Train the teacher of one cell, returns (teacher, acc, h_z, train_loader, test_loader).

    Resume training: if ckpt_dir is nonempty, save state_dict + acc + h_z after training; if resume=True and the
    cache exists, load it directly and skip training. The seed is set at the start of the function and each unit is independent, so
    "resumed run" and "continuous run" are unit-by-unit equivalent (the loaded weights replay the same distillation target).
    """
    torch.manual_seed(14000 + 100 * c_idx + s)
    (x_train, y_train), (x_test, y_test) = make_data(cell["arch"], n_per_class, s)
    train_loader = DataLoader(list(zip(x_train, y_train)), batch_size=128,
                              shuffle=True)
    test_loader = DataLoader(list(zip(x_test, y_test)), batch_size=256)
    teacher = build_model(cell["arch"], TEACHER_CODES, cell["metric"]).to(device)
    path = os.path.join(ckpt_dir, f"{cell['name']}_teacher_s{s}.pt") if ckpt_dir else None
    if resume and path and os.path.exists(path):
        ck = torch.load(path, map_location=device, weights_only=True)
        teacher.load_state_dict(ck["state_dict"])
        print(f"  seed {s}: [cache hit] teacher acc={ck['acc']:.3f}  H(Z_t)={ck['h_z']:.2f}")
        return teacher, ck["acc"], ck["h_z"], train_loader, test_loader
    opt = torch.optim.Adam(teacher.parameters(), lr=lr)
    for _ in range(epochs):
        train_one_epoch(teacher, train_loader, opt, device=device)
    acc_t, idx_t = evaluate(teacher, test_loader, device=device)
    h_z = entropy_of(idx_t[0], TEACHER_CODES)
    if path:
        torch.save({"state_dict": teacher.state_dict(), "acc": acc_t, "h_z": h_z}, path)
    return teacher, acc_t, h_z, train_loader, test_loader


def run_student_grid(cell, c_idx, teacher, k_list, epochs, lr, train_loader,
                     test_loader, s, device="cpu", ckpt_dir=None, resume=False):
    """Distillation grid (CE + KL(T=3)), returns {K: acc}. Seed scheme see pre-registration (6).

    Resume training: cache granularity = a single (cell, seed, K) unit, so an interruption loses at most one unit.
    """
    accs = {}
    for K in k_list:
        torch.manual_seed(15000 + 100 * c_idx + 100 * s + K)
        student = build_model(cell["arch"], K, cell["metric"]).to(device)
        path = (os.path.join(ckpt_dir, f"{cell['name']}_student_s{s}_K{K}.pt")
                if ckpt_dir else None)
        if resume and path and os.path.exists(path):
            accs[K] = float(torch.load(path, map_location="cpu",
                                       weights_only=True)["acc"])
            print(f"    K={K}: [cache hit] acc={accs[K]:.3f}")
            continue
        opt = torch.optim.Adam(student.parameters(), lr=lr)
        for _ in range(epochs):
            train_one_epoch(student, train_loader, opt, device=device,
                            distill_from=teacher)
        acc_s, _ = evaluate(student, test_loader, device=device)
        accs[K] = acc_s
        if path:
            torch.save({"acc": acc_s}, path)
    return accs


CALIBRATED_K = [8, 32, 48, 64, 96, 128, 192, 256]


def k_grid(k_min, quick):
    """The K grid (pure function, independently unit-testable).

    Non-quick uses the **fixed unified grid** CALIBRATED_K: all four cells share the same grid, eliminating
    the extra variable of "the grid adapting to K_min". Coverage check: the low-capacity end 8/32;
    near the Fano bound (measured K_min about 50-56) covered by 48/64; the cliff region covered by 96/128
    (measured K* in CP1/CP2 falls in 96-128); the plateau end covered by 192/256
    (the plateau criterion depends only on the top two levels, satisfied by 256/192). quick is only for the smoke test.
    """
    if quick:
        return sorted({8, 32, 64, 256, k_min})
    return list(CALIBRATED_K)


# ---------------------------------------------------------------------------
# Resume training (checkpoints)
# ---------------------------------------------------------------------------

CKPT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cp3")


def run_signature(quick, fast, seeds):
    """The config fingerprint of this run (pure function, independently unit-testable).

    The cache directory is stratified by fingerprint: different configs (quick/fast/full, the number of seeds, the epoch budget,
    the K grid, the teacher codebook size) do not contaminate each other -- otherwise a quick smoke test's cache would be
    misused by a full run, yielding wrong results.
    """
    ep = {a: cell_budget(a, quick)["epochs"] for a in ("mlp", "cnn")}
    n_per_class = 50 if quick else (100 if fast else 150)
    grid = "q" if quick else "-".join(str(k) for k in CALIBRATED_K)
    raw = "|".join([f"seeds{seeds}", f"npc{n_per_class}", f"mlpep{ep['mlp']}",
                    f"cnnep{ep['cnn']}", f"k{grid}", f"tc{TEACHER_CODES}"])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]


def ckpt_dir_for(quick, fast, seeds):
    """The cache directory of this run (created if it does not exist)."""
    d = os.path.join(CKPT_ROOT, run_signature(quick, fast, seeds))
    os.makedirs(d, exist_ok=True)
    return d


def verify_cp3(quick=False, fast=False, seeds=3, device="cpu", resume=False):
    """CP3 main flow: phase A (each cell teacher + registered prediction) -> phase B (distillation grid)
    -> key-cell hypothesis verdict + baseline-cell reproduction check.

    When resume=True, read checkpoints to resume training (see the "resume training" section); by default recompute and overwrite the cache.
    """
    n_per_class = 50 if quick else (100 if fast else 150)
    seeds = 1 if quick else max(1, seeds)
    ckpt_dir = ckpt_dir_for(quick, fast, seeds)
    print("\n" + "=" * 60)
    print(f"[CP3] metric intervention: 2x2 factor (architecture x VQ metric), {seeds} seeds/cell")
    print("=" * 60)
    print(f"  [checkpoint] {'resume mode (skip on cache hit)' if resume else 'recompute mode (overwrite cache)'}"
          f"  dir: {ckpt_dir}")
    print("  [motivation] CP1/CP2 twice failed to predict at the representation statistic level -> dissect the 'architecture family' composite variable.")
    print("  [pre-registered] H_metric(main): (MLP,cosine)=not_locatable, (CNN,euclid)=locatable")
    print("            H_architecture(competing): (MLP,cosine)=locatable,     (CNN,euclid)=not_locatable")
    print("            baseline cells also serve as reproduction check: (MLP,euclid) should be locatable; (CNN,cosine) should be not_locatable")
    print("            per-cell G0 double gate: teacher acc>=0.95 and H(Z_t)>=0.8*ln(K_t); "
          "each new cell's only variable vs its baseline cell = metric.")
    print("            cross-architecture claims forbidden (data modality/depth/BN/parameter count all change at once).")

    # ---- Phase A ----
    state = {}
    for c_idx, cell in enumerate(CELLS):
        budget = cell_budget(cell["arch"], quick)
        print(f"\n  --- cell {cell['name']} (arch={cell['arch']}, metric={cell['metric']}, "
              f"{budget['epochs']}ep, lr={budget['lr']}) ---")
        per_seed, teachers = [], []
        for s in range(seeds):
            teacher, acc, h_z, tr, te = train_teacher(
                cell, c_idx, s, budget["epochs"], n_per_class, budget["lr"], device,
                ckpt_dir=ckpt_dir, resume=resume)
            g0, ok_acc, ok_cb = g0_gate(acc, h_z, TEACHER_CODES)
            g0_tag = "PASS" if g0 else f"FAIL acc{'PASS' if ok_acc else 'FAIL'}/codebook{'PASS' if ok_cb else 'collapsed'}"
            r_ib = task_info_fano(1.0 - acc, 64)
            print(f"  seed {s}: teacher acc={acc:.3f} (G0 {g0_tag})  "
                  f"H(Z_t)={h_z:.2f}  R_IB(Fano)={r_ib:.2f} nats")
            per_seed.append(dict(acc=acc, h_z=h_z, g0=g0, r_ib=r_ib,
                                 k_min=max(2, math.ceil(math.exp(r_ib)))))
            teachers.append((teacher, acc, tr, te))
        state[cell["name"]] = dict(cell=cell, c_idx=c_idx, budget=budget,
                                   per_seed=per_seed,
                                   g0_all=all(r["g0"] for r in per_seed),
                                   teachers=teachers)

    # Registered prediction (before any student data)
    print("\n  --- phase A done: registered predictions (before phase B data) ---")
    mp, ap = metric_prediction(), arch_prediction()
    for k in KEY_CELLS:
        print(f"  {k}: H_metric -> {mp[k]} ; H_architecture -> {ap[k]}")

    # ---- Phase B ----
    print("\n  --- phase B: per-cell distillation grid (gap curve + plateau preflight) ---")
    for cell in CELLS:
        st = state[cell["name"]]
        print(f"\n  --- cell {cell['name']} (metric={cell['metric']}) ---")
        records = []
        for s, (teacher, acc_t, tr, te) in enumerate(st["teachers"]):
            k_list = k_grid(st["per_seed"][s]["k_min"], quick)
            accs = run_student_grid(cell, st["c_idx"], teacher, k_list,
                                    st["budget"]["epochs"], st["budget"]["lr"],
                                    tr, te, s, device,
                                    ckpt_dir=ckpt_dir, resume=resume)
            plateau = distill_plateau_ok(k_list, accs)
            k_star = distill_gap_k_star(acc_t, k_list, accs) if plateau else None
            records.append((plateau, k_star))
            gaps = {K: acc_t - accs[K] for K in k_list}
            print(f"  seed {s}: teacher acc={acc_t:.3f}  "
                  f"R_IB(Fano)={st['per_seed'][s]['r_ib']:.2f} -> K_min="
                  f"{st['per_seed'][s]['k_min']}")
            print("    gap g(K): " + "  ".join(f"{K}:{gaps[K]:.3f}" for K in k_list))
            print(f"    plateau established={'yes' if plateau else 'no'}; K*(gap)={k_star}")
        st["actual"] = locatability(records) if st["g0_all"] else None
        print(f"  [cell measured] {st['actual'] if st['actual'] is not None else 'none (G0 not passed, cell invalid)'}")

    if quick:
        print("\n  [smoke] quick mode outputs no verdict.")
        return

    # ---- Verdict ----
    actual = {c["name"]: state[c["name"]]["actual"] for c in CELLS}
    print("\n  --- reproduction check (baseline cells) ---")
    for name, expect in (("MLP-euclid", "locatable"), ("CNN-cosine", "not_locatable")):
        got = actual[name]
        tag = "reproduction OK" if got == expect else ("reproduction FAILURE" if got is not None else "cell invalid")
        print(f"  {name}: expected {expect}  measured {got if got is not None else 'none'}  -> {tag}")

    print("\n  --- key-cell hypothesis verdict ---")
    verdict, detail = hypothesis_verdict(actual)
    print(f"  {verdict}")
    for line in detail:
        print(line)
    print("\n  [statement] dose/cell are not independent (within the same architecture/task only the metric varies) -- the causal "
          "role of the new cells supporting the metric holds only within each architecture; no cross-architecture claim.")


def main():
    ap = argparse.ArgumentParser(description="CP3: metric intervention (2x2 factor)")
    ap.add_argument("--quick", action="store_true", help="smoke (no verdict)")
    ap.add_argument("--fast", action="store_true", help="reduced budget but with verdict")
    ap.add_argument("--seeds", type=int, default=3,
                    help="seeds per cell (default 3; quick fixed at 1)")
    ap.add_argument("--resume", action="store_true",
                    help="resume training: skip (cell,seed[,K]) units on cache hit")
    args = ap.parse_args()
    verify_cp3(quick=args.quick, fast=args.fast, seeds=args.seeds,
               resume=args.resume)
    print("\nDone.")


if __name__ == "__main__":
    main()
