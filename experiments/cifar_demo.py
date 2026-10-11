"""CIFAR-10 + CNN validation experiment: generalization of the capacity lens to convolutional networks and harder tasks.

Usage:
    python experiments/cifar_demo.py              # all (about 60 minutes on CPU)
    python experiments/cifar_demo.py --quick      # quick smoke test (subset + short training, no verdict)
    python experiments/cifar_demo.py --only c2    # run a single experiment
    python experiments/cifar_demo.py --only c3    # C3 v3: synthetic-image version grid64 (current protocol)
    python experiments/cifar_demo.py --only c3-cifar  # C3 legacy version (CIFAR-10, archived)

Division of labor with the MNIST line:
- MNIST+MLP validates the mechanism (the controlled form of P1/P2/P3);
- this line validates the real deployment scenario: the bottleneck inserted after the convolutional backbone, on a harder task (higher R_IB).
The decision criteria are given in the verdict functions of each experiment in this script.

Data placement (the user copies manually, the tool does not download):
    experiments/data/cifar-10-batches-py/          # extracted folder (containing data_batch_1..5 etc.)
    or experiments/data/cifar-10-python.tar.gz     # archive (torchvision will extract it)
"""

import argparse
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
from capacity_lens import (VQBottleneck, FisherEffectiveDim,
                           train_one_epoch, evaluate, entropy_of)
from capacity_lens.training import task_info_fano
from mnist_demo import (distill_plateau_ok, distill_gap_k_star, ratio_verdict,
                        g0_gate)

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")

# Data placement (verified directory layout, the tool does not download from the network):
#   experiments/data/cifar-10-batches-py/data_batch_1 ... test_batch (extracted folder)
# Note: what torchvision verifies is the md5 of this extracted folder; download=True returns
#       immediately when the files already exist (offline-safe, no network calls in practice).
#       A cifar-10-python.tar.gz backup may be kept alongside; it is
#       not required (can be deleted to save 162MB).


# ---------------------------------------------------------------------------
# Data and models
# ---------------------------------------------------------------------------

def get_cifar10(batch_size=128, subset=None):
    """CIFAR-10. subset: fraction of the training set to keep (used in quick mode)."""
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=train_tf)
    test = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=test_tf)
    if subset is not None:
        n_keep = int(len(train) * subset)
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(train), generator=g)[:n_keep]
        train = Subset(train, idx.tolist())
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class SmallCNN(nn.Module):
    """Convolutional backbone + VQ bottleneck + linear head (about 140k parameters, trainable on CPU).

    Capacity lens insertion point: the d-dimensional representation after global pooling/flattening
    (in real deployments it is exactly this kind of location that quantization/distillation act on).
    d is the compression projection width and K is the codebook size -- the two are
    mutually independent capacity knobs (C1 sweeps K, C2 sweeps d).
    """

    def __init__(self, d=64, n_codes=256, width=32):
        super().__init__()
        w2 = width * 2
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, w2, 3, padding=1), nn.BatchNorm2d(w2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(w2, w2, 3, padding=1), nn.BatchNorm2d(w2), nn.ReLU(), nn.MaxPool2d(2),
        )  # 32 -> 4, output [B, w2, 4, 4]
        self.fc1 = nn.Linear(w2 * 4 * 4, d)
        self.vq = VQBottleneck(d, n_codes)
        self.fc2 = nn.Linear(d, 10)

    def forward(self, x):
        h = self.features(x).flatten(1)
        t = F.relu(self.fc1(h))
        z, idx = self.vq(t)
        return self.fc2(z), [idx]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Synthetic-image version grid64 (C3 v3 task and model; encoding parameters pre-registered, do not tune)
# ---------------------------------------------------------------------------

def make_grid64_images(n_per_class=150, sigma_pix=0.15, sigma_bright=0.1, seed=0):
    """Synthetic-image version grid64: 16x16 single channel, three bands x 4 block positions -> 64 classes.

    H(Y) = ln 64 = 4.16 nats (y is sampled uniformly, guaranteed by construction). The encoding
    parameters match the C3 v3 pre-registration (2026-09-19), do not tune them -- tuning the noise/
    brightness to accommodate the results = p-hacking.

    Encoding: three horizontal bands (top row of the 3x3 block at 1/7/12); within band c the block
    center column x in {2,6,10,14} encodes nu_c in {0..3}; the block brightness is ~N(1.0, sigma_bright)
    (a task-irrelevant style variation); whole-image pixel noise is N(0, sigma_pix).
    Design rationale: a positional encoding is safe with respect to BatchNorm (a "global-intensity
    encoding" would be erased by per-channel normalization); the block is a natural response target
    for a 3x3 convolution -- it does not fight the convolutional inductive bias.
    """
    g = torch.Generator().manual_seed(seed)
    n_classes = 64
    rows0 = (1, 7, 12)          # top row of the 3x3 block in each band
    cols = (2, 6, 10, 14)       # center column of the 4 positions
    templates = torch.zeros(n_classes, 1, 16, 16)
    for v in range(n_classes):
        v1, rem = divmod(v, 16)
        v2, v3 = divmod(rem, 4)
        for c, vi in enumerate((v1, v2, v3)):
            r0, cx = rows0[c], cols[vi]
            templates[v, 0, r0:r0 + 3, cx - 1:cx + 2] = 1.0
    y = torch.arange(n_classes).repeat_interleave(n_per_class)
    bright = 1.0 + sigma_bright * torch.randn(len(y), 1, 1, 1, generator=g)
    x = templates[y] * bright + sigma_pix * torch.randn(len(y), 1, 16, 16, generator=g)
    perm = torch.randperm(len(y), generator=g)
    x, y = x[perm], y[perm]
    n_train = int(0.8 * len(y))
    return (x[:n_train], y[:n_train]), (x[n_train:], y[n_train:])


class GridCNN(nn.Module):
    """CNN for the synthetic-image version grid64: 1x16x16 -> 3x(conv+BN+pool) -> fc1->d -> VQ(K) -> 64 classes.

    Adapted version of SmallCNN (1-channel input, 16x16, 64-class output, about 50k parameters).
    The only capacity knob is K: the student varies only K (d fixed at 64, isomorphic to the teacher
    -- consistent with "sweep only the codebook size" of Function B v3).
    """

    def __init__(self, d=64, n_codes=256, width=32, metric="cosine"):
        super().__init__()
        w2 = width * 2
        self.features = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, w2, 3, padding=1), nn.BatchNorm2d(w2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(w2, w2, 3, padding=1), nn.BatchNorm2d(w2), nn.ReLU(), nn.MaxPool2d(2),
        )  # 16 -> 2, output [B, w2, 2, 2]
        self.fc1 = nn.Linear(w2 * 2 * 2, d)
        self.vq = VQBottleneck(d, n_codes, metric=metric)
        self.fc2 = nn.Linear(d, 64)

    def forward(self, x):
        h = self.features(x).flatten(1)
        t = F.relu(self.fc1(h))
        z, idx = self.vq(t)
        return self.fc2(z), [idx]


# ---------------------------------------------------------------------------
# C1: log K tightness (CNN version)
# ---------------------------------------------------------------------------

def verify_c1(device="cpu", d=64, epochs=3, quick=False, fast=False):
    """Fixed width d, sweep the codebook size K, measure whether H(Z) grows with log K (CNN version of P1).

    Criterion: last-segment slope >= 0.7 passes; < 0.5 with utilization < 30% fails (collapse).
    """
    k_list = (16, 64, 256)
    subset = 0.2 if (quick or fast) else None
    if quick:
        epochs = 1
    elif fast:
        epochs = 2
    print("\n" + "=" * 60)
    print(f"[C1] log K tightness (SmallCNN, CIFAR-10, d={d})")
    print("=" * 60)

    train_loader, test_loader = get_cifar10(subset=subset)
    results = []
    for K in k_list:
        model = SmallCNN(d=d, n_codes=K).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(epochs):
            train_one_epoch(model, train_loader, opt, device=device)
        acc, idx_lists = evaluate(model, test_loader, device=device)
        h_z = entropy_of(idx_lists[0], K)
        util = model.vq.codebook_utilization(idx_lists[0])
        log_k = math.log(K)
        results.append((K, log_k, h_z, acc, util))
        print(f"  K={K:4d}  log K={log_k:5.2f}  H(Z)={h_z:5.2f}  "
              f"H(Z)/logK={h_z/log_k:5.2f}  util={util:4.0%}  acc={acc:.3f}")

    if not quick and len(results) >= 2:
        (_, l0, h0, _, u0), (_, l1, h1, _, u1) = results[-2], results[-1]
        slope = (h1 - h0) / (l1 - l0)
        print(f"  [C1 criterion] last-segment slope = {slope:.2f} (toward 1 = tight)")
        if slope >= 0.7:
            verdict = "PASS: the upper bound is equally tight on CNN"
        elif slope < 0.5 and (u0 < 0.3 or u1 < 0.3):
            verdict = "FAIL: codebook collapse (cosine VQ fails on CNN representations; consider euclid/EMA)"
        else:
            verdict = "PARTIAL: loose but no collapse"
        print(f"  [C1 verdict] {verdict}")
    return results


# ---------------------------------------------------------------------------
# C2: bottleneck-width capacity ceiling (CNN version of P3)
# ---------------------------------------------------------------------------

def verify_c2(device="cpu", k_fixed=256, epochs=3, quick=False, fast=False):
    """Fixed K, sweep the compression projection width d, measure test accuracy under a fixed budget.

    Unlike P3 (the convergence-speed form), here we measure the "capacity ceiling" form:
    d too small -> insufficient end-to-end information for linear compression + quantization -> accuracy hits a wall.
    Criteria:
    - success: acc(d_max) - acc(d_min) >= 0.10 (a clear capacity wall) and roughly monotonically increasing;
    - failure: acc(d_min) >= acc(d_max) - 0.05 (small d costs nothing, the width claim is falsified).
    """
    if quick:
        d_list, epochs, subset = (4, 16, 64), 1, 0.2
    elif fast:
        d_list, epochs, subset = (4, 16, 64), 2, 0.2
    else:
        d_list, subset = (4, 8, 16, 32, 64), None
    print("\n" + "=" * 60)
    print(f"[C2] bottleneck-width capacity ceiling (K={k_fixed}, fixed {epochs} epoch budget)")
    print("=" * 60)

    train_loader, test_loader = get_cifar10(subset=subset)
    results = []
    for d in d_list:
        model = SmallCNN(d=d, n_codes=k_fixed).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(epochs):
            train_one_epoch(model, train_loader, opt, device=device)
        acc, idx_lists = evaluate(model, test_loader, device=device)
        h_z = entropy_of(idx_lists[0], k_fixed)
        results.append((d, acc, h_z))
        print(f"  d={d:3d}  acc={acc:.3f}  H(Z)={h_z:.2f}")

    if not quick and len(results) >= 3:
        accs = [a for _, a, _ in results]
        wall = accs[-1] - accs[0]
        rising = all(accs[i + 1] >= accs[i] - 0.02 for i in range(len(accs) - 1))
        print(f"  [C2 criterion] acc(d={d_list[-1]}) - acc(d={d_list[0]}) = {wall:.3f}")
        if wall >= 0.10 and rising:
            verdict = "PASS: small d hits the capacity wall; the width claim holds on CNN"
        elif wall < 0.05:
            verdict = "FAIL: small d costs almost nothing (width is not a bottleneck)"
        else:
            verdict = f"PARTIAL: wall height {wall:.3f} in 0.05-0.10" + ("" if rising else ", and non-monotonic")
        print(f"  [C2 verdict] {verdict}")
    return results


# ---------------------------------------------------------------------------
# C3: distillation capacity lower bound (CNN version)
# ---------------------------------------------------------------------------

def verify_c3_cifar(device="cpu", d=64, k_teacher=256, epochs_teacher=3, epochs_student=None,
                    quick=False, fast=False):
    """C3 legacy version (CIFAR-10) -- **archived**: the pre-probe (2026-09-19) killed CIFAR-10
    (the information content of 10 classes is insufficient to form a wall: teacher acc=0.559 -> Fano K_min=2, hard ceiling 10).
    Kept only to reproduce the historical record; the current protocol is verify_c3_v3 (--only c3).

    teacher CNN -> students with different K distilled, verifying K_min = ceil(e^R_IB) (CNN version of Function B).

    Criterion: the practically smallest usable student K* (acc >= teacher - 0.03)
    satisfies K*/2 <= K_min <= 2K* (one octave).

    Budget alignment (fixed 2026-09-19): the student's number of epochs defaults to the teacher's, otherwise the student
    systematically underfits, K* is pushed up and the criterion necessarily fails (previously student=2 < teacher=3 was that defect).
    R_IB uses H(Z_teacher) (the representation rate, the quantity transferred by distillation); see mnist_demo.verify_distill for the reasoning.
    """
    subset = 0.2 if (quick or fast) else None
    if quick:
        epochs_teacher = 1
    elif fast:
        epochs_teacher = 2
    if epochs_student is None:
        epochs_student = epochs_teacher  # budget alignment
    print("\n" + "=" * 60)
    print("[C3] distillation capacity lower bound K_min = ceil(e^R_IB) (SmallCNN, CIFAR-10)")
    print("=" * 60)

    train_loader, test_loader = get_cifar10(subset=subset)
    teacher = SmallCNN(d=d, n_codes=k_teacher).to(device)
    opt = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    for _ in range(epochs_teacher):
        train_one_epoch(teacher, train_loader, opt, device=device)
    acc_t, idx_t = evaluate(teacher, test_loader, device=device)
    r_ib = entropy_of(idx_t[0], k_teacher)
    k_min = max(2, math.ceil(math.exp(r_ib)))
    print(f"  [teacher] acc={acc_t:.3f}  R_IB={r_ib:.2f} nats -> K_min = {k_min}")

    k_list = (16, 64, 256) if quick else (16, 32, 64, 128, 256)
    results = []
    print(f"  {'K':>4} {'acc':>7} {'OK':>4}")
    for K in k_list:
        student = SmallCNN(d=d, n_codes=K).to(device)
        opt = torch.optim.Adam(student.parameters(), lr=1e-3)
        for _ in range(epochs_student):
            train_one_epoch(student, train_loader, opt, device=device, distill_from=teacher)
        acc_s, _ = evaluate(student, test_loader, device=device)
        ok = acc_s >= acc_t - 0.03
        results.append((K, acc_s, ok))
        print(f"  {K:>4} {acc_s:7.3f} {'Y' if ok else 'N':>4}")

    if not quick:
        usable = [k for k, a, ok in results if ok]
        if usable:
            k_star = min(usable)
            lo, hi = k_star / 2, k_star * 2
            print(f"  [C3 criterion] actual smallest usable student K*={k_star}, predicted K_min={k_min}, octave window [{lo:.0f}, {hi}]")
            if lo <= k_min <= hi:
                verdict = "PASS: K_min falls within one octave of the actual cliff (also holds on CNN)"
            else:
                verdict = f"FAIL: K_min deviates from K* by more than one octave ({'too conservative' if k_min > hi else 'too aggressive'})"
            print(f"  [C3 verdict] {verdict}")
        else:
            print("  [C3 verdict] no student meets the bar (insufficient budget or distillation config issue, UNDECIDABLE)")
    return r_ib, k_min, results


def verify_c3_v3(quick=False, fast=False, seeds=3, k_teacher=256, device="cpu"):
    """C3 v3: distillation capacity lower bound for the synthetic-image version grid64 (CNN architecture).

    Protocol pre-registration (2026-09-19, criteria before data).
    - Task make_grid64_images: 16x16 single channel, three bands x 4 block positions -> 64 classes.
      H(Y)=ln64=4.16 nats is guaranteed by construction; when student K<64, logK<H(Y), and
      DPI+Fano mathematically forbid preserving accuracy -- the sharpness of the cliff is guaranteed by the task structure (same as Function B v3).
    - The only difference from Function B v3: the input is an image and the model is a convolutional network (GridCNN).
      What is verified is whether the tool holds on a CNN architecture + image input.
    - R_IB uses the Fano lower bound (v3 convention, unified with Function A/B).

    Three layers of criteria: G0 pre-gate (teacher acc >= 0.95 and H(Z_t) >= 0.8*ln(K_t);
    the second one was added 2026-09-22, motivation see mnist_demo.g0_gate -- this experiment is exactly the victim
    case let through by codebook collapse, whose "negative result" turned out to be a budget artifact upon CP3 review);
    the main criterion (K* falls within the octave of K_min + plateau establishment);
    hard-wall directionality (gap column R_IB-logK: positive at K=32,
    turning negative around K=64).
    """
    epochs = 3 if quick else (6 if fast else 10)
    n_per_class = 50 if quick else (100 if fast else 150)
    lr = 1e-3  # CNN convention for Adam+BN (the v3 linear model's 1e-2 does not apply here)
    seeds = 1 if quick else max(1, seeds)
    wall = 64
    print("\n" + "=" * 60)
    print(f"[C3 v3] distillation capacity lower bound K_min = ceil(e^R_IB) (synthetic-image grid64 + CNN; {seeds} seeds)")
    print("=" * 60)
    print("  [background] the pre-probe ruled out CIFAR-10 (10 classes, Fano K_min=2, hard ceiling 10). v3 switches to synthetic")
    print("        image version grid64: H(Y)=ln64=4.16 nats, wall depth guaranteed by the task structure.")
    print("  [R_IB] Fano lower bound of I(Z_t;Y) (from the teacher error rate, unified with Function A/B).")
    print("  [criterion] G0 double gate: teacher acc>=0.95 and H(Z_t)>=0.8*ln(K_t); "
          "the gap curve locates K* (plateau preflight);")
    print("             octave criterion + hard-wall directionality (gap column R_IB-logK).")

    records, curves, k_list = [], [], None
    h_zs = []   # teacher representation entropy (used for the G0 codebook gate; same order as curves)
    for s in range(seeds):
        # both the data realization and the initialization vary with the seed (the seed changes only the randomness, not the nominal task information ln 64)
        torch.manual_seed(6000 + s)
        (x_train, y_train), (x_test, y_test) = make_grid64_images(
            n_per_class=n_per_class, seed=s)
        train_loader = DataLoader(list(zip(x_train, y_train)), batch_size=128, shuffle=True)
        test_loader = DataLoader(list(zip(x_test, y_test)), batch_size=256)

        # 1. teacher (K=256, isomorphic to the student, budget strictly aligned)
        teacher = GridCNN(d=64, n_codes=k_teacher)
        opt = torch.optim.Adam(teacher.parameters(), lr=lr)
        for _ in range(epochs):
            train_one_epoch(teacher, train_loader, opt, device=device)
        acc_t, idx_t = evaluate(teacher, test_loader, device=device)
        h_z_t = entropy_of(idx_t[0], k_teacher)  # printed for comparison: representation entropy vs task information
        r_ib = task_info_fano(1.0 - acc_t, 64, h_y=math.log(64))
        k_min = max(2, math.ceil(math.exp(r_ib) - 1e-9))

        # 2. student grid: anchored at K_min of seed 0; the fine steps 48-96 ensure the knee is visible
        if k_list is None:
            if quick:
                k_list = sorted({8, 32, 64, 256, k_min})
            else:
                k_list = sorted({8, 16, 32, 48, 56, 64, 72, 96, 128, 192, 256,
                                 k_min, max(2, k_min - 2), min(k_teacher, k_min + 2)})
            print(f"  [K grid] anchored at K_min={k_min} of seed 0 (densified near the hard wall K=64): {k_list}")

        # 3. distillation of a student for each K (CE + KL(T=3), budget strictly aligned with the teacher)
        accs = {}
        for K in k_list:
            torch.manual_seed(7000 + 100 * s + K)
            student = GridCNN(d=64, n_codes=K)
            opt = torch.optim.Adam(student.parameters(), lr=lr)
            for _ in range(epochs):
                train_one_epoch(student, train_loader, opt, device=device,
                                distill_from=teacher)
            acc_s, _ = evaluate(student, test_loader, device=device)
            accs[K] = acc_s

        plateau = distill_plateau_ok(k_list, accs)
        k_star = distill_gap_k_star(acc_t, k_list, accs) if plateau else None
        records.append((s, k_min, k_star))
        curves.append((s, acc_t, k_min, accs))
        h_zs.append(h_z_t)

        gaps = {K: acc_t - accs[K] for K in k_list}
        g_ref = gaps[k_list[-1]]
        print(f"  seed {s}: teacher acc={acc_t:.3f}  R_IB(Fano)={r_ib:.2f} nats -> K_min={k_min}"
              f"  [cf. H(Z_t)={h_z_t:.2f} nats]")
        print("    gap g(K): " + "  ".join(f"{K}:{gaps[K]:.3f}" for K in k_list))
        print("    gap column R_IB-logK: " + "  ".join(f"{K}:{r_ib - math.log(K):+.2f}" for K in k_list)
              + " (positive = insufficient information)")
        print(f"    plateau established={'Y' if plateau else 'N'}; **K*(gap)={k_star}**"
              f"; [hard wall] K={wall}; g_ref=g({k_list[-1]})={g_ref:.3f}")

    # per-K x per-seed gap matrix (isomorphic to Function B v3)
    heads = " ".join(f"{'seed' + str(s):>9}" for s, _, _, _ in curves)
    print(f"\n  gap matrix g(K)=teacher-acc  {'K':>4} {heads} {'mean':>9}")
    for K in k_list:
        row = " ".join(f"{acc_t - accs[K]:9.3f}" for _, acc_t, _, accs in curves)
        mean = sum(acc_t - accs[K] for _, acc_t, _, accs in curves) / len(curves)
        print(f"  {'':>22} {K:>4} {row} {mean:9.3f}")
    print("  teacher acc per seed: " + ", ".join(f"seed{s}={a:.3f}" for s, a, _, _ in curves))

    if not quick:
        gates = [g0_gate(a, h, k_teacher) for (_, a, _, _), h in zip(curves, h_zs)]
        g0_all = all(g[0] for g in gates)
        if g0_all:
            print("  [G0 gate] teacher acc>=0.95 and H(Z_t)>=0.8*ln(K_t): PASS")
        else:
            n_acc_bad = sum(1 for g in gates if not g[1])
            n_cb_bad = sum(1 for g in gates if not g[2])
            print(f"  [G0 gate] FAIL (acc below bar on {n_acc_bad} seed(s); "
                  f"codebook collapse on {n_cb_bad} seed(s)) (verdict invalid)")
        if seeds > 1:
            print(f"\n  [C3 multi-seed summary] {seeds} seeds (criterion: gap curve, synthetic-image grid64)")
        verdict, detail = ratio_verdict(records, "measured K*(gap) / predicted K_min")
        for line in detail:
            print(line)
        print(f"  [C3 v3 verdict] {verdict}" + ("" if g0_all else " (G0 gate FAIL, verdict invalid)"))
    return records, curves


# ---------------------------------------------------------------------------
# C4: Fisher effective dimension curve (CNN version)
# ---------------------------------------------------------------------------

def fisher_demo_cnn(device="cpu", d=64, k=256, epochs=3, quick=False, fast=False):
    """v1 acceptance (CNN): the d_eff curve is finite and nonzero (hard criterion); whether it contracts during training (soft observation)."""
    subset = 0.2 if (quick or fast) else None
    if quick:
        epochs = 1
    elif fast:
        epochs = 2
    print("\n" + "=" * 60)
    print("[C4] Fisher effective dimension d_eff (SmallCNN, CIFAR-10)")
    print("=" * 60)

    train_loader, test_loader = get_cifar10(subset=subset)
    model = SmallCNN(d=d, n_codes=k).to(device)
    fed = FisherEffectiveDim(model, max_batches=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    curve = []
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out, _ = model(x)
            loss = F.cross_entropy(out, y)
            for m in model.modules():
                if isinstance(m, VQBottleneck):
                    loss = loss + 0.25 * m.commit_loss()
            opt.zero_grad()
            loss.backward()
            fed.collect()
            opt.step()
        d_eff = fed.effective_dim()
        fed.reset()
        acc, _ = evaluate(model, test_loader, device=device)
        curve.append((ep, d_eff, acc))
        print(f"  epoch {ep}: d_eff = {d_eff:6.2f}  acc = {acc:.3f}")

    if not quick and len(curve) >= 2:
        finite = all(de is not None and 0 < de <= 32 for _, de, _ in curve)
        first, last = curve[0][1], curve[-1][1]
        trend = "contraction (entering a low-dim basin)" if last < first else "no contraction"
        print(f"  [C4 criterion] curve finite and nonzero: {'Y' if finite else 'N (bug)'}")
        print(f"  [C4 observation] d_eff first/last {first:.2f} -> {last:.2f}: {trend}")
        print(f"  [C4 verdict] {'PASS (engineering acceptance)' if finite else 'FAIL: numerical anomaly'}; trend: {trend}")
    return curve


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Capacity Lens CIFAR/CNN validation experiment")
    ap.add_argument("--quick", action="store_true", help="quick smoke mode (20%% subset + 1 epoch, no verdict)")
    ap.add_argument("--fast", action="store_true",
                    help="reduced budget mode (20%% subset + 2 epochs), **still outputs a verdict**")
    ap.add_argument("--only", choices=["c1", "c2", "c3", "c3-cifar", "c4"], default=None,
                    help="run only the given experiment (c3 = v3 synthetic-image grid64; c3-cifar = archived legacy, ruled out by the pre-probe)")
    args = ap.parse_args()
    quick = args.quick
    fast = args.fast and not quick

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode = "quick (smoke, no verdict)" if quick else ("fast (reduced budget, verdict)" if fast else "full (complete validation)")
    print(f"device: {device}  mode: {mode}")

    if args.only in (None, "c1"):
        verify_c1(device=device, quick=quick, fast=fast)
    if args.only in (None, "c2"):
        verify_c2(device=device, quick=quick, fast=fast)
    if args.only in (None, "c3"):
        verify_c3_v3(quick=quick, fast=fast)
    if args.only == "c3-cifar":
        verify_c3_cifar(device=device, quick=quick, fast=fast)
    if args.only in (None, "c4"):
        fisher_demo_cnn(device=device, quick=quick, fast=fast)

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
