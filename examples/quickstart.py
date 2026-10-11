"""Quickstart: the three numbers Capacity Lens produces, on a task it builds itself.

Runs in a minute or two on a laptop CPU (seconds on a GPU) and needs no dataset and no repository-internal
imports -- everything comes from the installed `capacity_lens` package, which is itself part of the
demonstration: the library is self-sufficient.

    python examples/quickstart.py

What it prints, and how to read it:

  1. R_IB and K_min -- the task's information requirement, read from a REFERENCE model's achieved error
     (Fano). No bottleneck is needed for this: any model that scores well on the task is a witness.
  2. the certified exclusion -- every K with log K < R_IB provably cannot carry the task. This direction
     is SOUND (it comes from a lower bound); it is also conservative.
  3. the practical wall K_wall and the looseness -- the smallest student that still matches the reference,
     and how far below it the certified bound sits. `K_wall` is what a sizing decision should use; the
     looseness tells you whether the certified bound is worth quoting at all.

The task: C = 27 classes on a 3-D lattice (the project's own synthetic family). Both the reference and the
students are the same tiny MLP, so the only thing that varies is the bottleneck.
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Run straight from a checkout (`python examples/quickstart.py`) as well as after `pip install -e .`:
# this is a location bootstrap only -- the script imports nothing from experiments/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capacity_lens import (VQBottleneck, DistillCapacity, task_info_fano, train_one_epoch, evaluate,
                           entropy_of, excluded_set, coverage, false_exclusions, measured_wall,
                           looseness, preflight_report, wall_geometry_probe)

SEED = 0
GRID = 3                      # 3x3x3 = 27 classes
SAMPLES_PER_CLASS = 150
SIGMA = 0.5                   # lattice noise: the difficulty knob (0.5 keeps the wall locatable)
SPACING = 3.0
EPOCHS = 40
LR = 1e-2
D_REP = 64
K_TEACHER = 256               # the reference's own bottleneck (kept large enough not to be binding)
STUDENT_GRID = [2, 4, 8, 16, 32, 64, 128, 256]


def make_lattice(grid=GRID, samples_per_class=SAMPLES_PER_CLASS, sigma=SIGMA, spacing=SPACING, seed=SEED):
    """C = grid**3 Gaussian classes on a lattice; returns ((x_tr, y_tr), (x_te, y_te))."""
    g = torch.Generator().manual_seed(seed)
    coords = torch.cartesian_prod(*[torch.arange(grid) for _ in range(3)]).float()
    centers = coords * spacing
    y = torch.arange(len(centers)).repeat_interleave(samples_per_class)
    x = centers[y] + sigma * torch.randn(len(y), 3, generator=g)
    perm = torch.randperm(len(y), generator=g)
    x, y = x[perm], y[perm]
    n_train = int(0.8 * len(y))
    return (x[:n_train], y[:n_train]), (x[n_train:], y[n_train:])


class BottleneckMLP(nn.Module):
    """Linear(3->d) -> ReLU -> VQ(d, K) -> Linear(d->C): the capacity knob is K."""

    def __init__(self, n_classes, k_codes, d=D_REP):
        super().__init__()
        self.fc1 = nn.Linear(3, d)
        self.vq = VQBottleneck(d, k_codes, metric="euclid")
        self.fc2 = nn.Linear(d, n_classes)

    def forward(self, x):
        z, idx = self.vq(F.relu(self.fc1(x)))
        return self.fc2(z), [idx]


def train_model(x_tr, y_tr, x_te, y_te, n_classes, k_codes, seed, device, teacher=None):
    """Train one unit (reference when k_codes is None) and return (accuracy, model, codebook entropy)."""
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(seed)                          # seed BEFORE construction: init is then deterministic
    if k_codes is None:
        model = nn.Sequential(nn.Linear(3, D_REP), nn.ReLU(), nn.Linear(D_REP, n_classes)).to(device)
    else:
        model = BottleneckMLP(n_classes, k_codes).to(device)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=128, shuffle=True)
    test = DataLoader(TensorDataset(x_te, y_te), batch_size=512)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(EPOCHS):
        train_one_epoch(model, loader, opt, device=device, distill_from=teacher)
    acc, idx_lists = evaluate(model, test, device=device)
    h_z = entropy_of(idx_lists[0], k_codes) if idx_lists else None
    return acc, model, h_z


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = GRID ** 3
    print(f"Capacity Lens quickstart | device={device} | C={n_classes} classes on a 3-D lattice\n")

    (x_tr, y_tr), (x_te, y_te) = make_lattice()

    # --- 1. the reference (a witness) -> the requirement -------------------------------
    # For the READING alone a bottleneck-free reference is enough; here it is given a large VQ so that the
    # students below are distilled from the same model family (see docs/class-count-axis.md on why
    # that matters for measuring a wall).
    acc_t, teacher, h_z_t = train_model(x_tr, y_tr, x_te, y_te, n_classes, K_TEACHER, 1, device)
    err = 1.0 - acc_t
    r_ib = task_info_fano(err, n_classes)
    k_min = DistillCapacity().min_codebook_size(r_ib)
    print(f"[1] reference: acc={acc_t:.3f} (err={err:.3f})  H(Z_t)={h_z_t:.2f} nats")
    print(f"    requirement R_IB = {r_ib:.2f} nats  ->  K_min = ceil(e^R_IB) = {k_min}")
    print(f"    (the requirement needs ONLY this accuracy: no bottleneck and no student training)\n")

    # --- 2. the certified exclusion (zero training) ------------------------------------
    certified = excluded_set(r_ib, STUDENT_GRID)
    print(f"[2] certified exclusion (log K < R_IB): {certified or 'none on this grid'}")
    print(f"    sound but conservative: anything it excludes is provably infeasible,")
    print(f"    anything it keeps is merely not-yet-ruled-out\n")

    # --- 3. students -> the practical wall and the looseness ---------------------------
    accs = {}
    for K in STUDENT_GRID:
        accs[K], _, _ = train_model(x_tr, y_tr, x_te, y_te, n_classes, K, 100 + K, device, teacher)
        print(f"    student K={K:4d}: acc={accs[K]:.3f}  relative gap={(acc_t - accs[K]) / acc_t:+.2f}")
    k_wall, rel = measured_wall(STUDENT_GRID, accs, acc_t, rel_tol=0.10)
    lo = looseness(r_ib, k_wall)
    fails = sorted(K for K in accs if rel[K] >= 0.50)
    succeeds = sorted(K for K in accs if rel[K] <= 0.10)
    print(f"\n[3] practical wall K_wall = {k_wall} (the smallest K matching the reference within 10%)")
    print(f"    looseness = ln(K_wall)/R_IB = {None if lo is None else round(lo, 2)}"
          f"   (1.0 = the bound is the whole answer; >> 1 = the bound is only a floor)")
    print(f"    measured: succeeds={succeeds} fails={fails} "
          f"| exclusion coverage={coverage(certified, fails):.2f} "
          f"| false exclusions={false_exclusions(certified, succeeds)}")

    # The wall probe belongs on the reference's REPRESENTATION (that is what K* would be searched in),
    # and the accuracy gate is task-specific: this synthetic task with sigma=0.5 reaches ~0.97, whereas
    # the project's CIFAR-10 notes used 0.78 -- thresholds are not universal.
    with torch.no_grad():
        feats = F.relu(teacher.fc1(x_tr.to(device))).cpu()
    probe = wall_geometry_probe(feats, y_tr)
    report = preflight_report(acc_t, h_z_t or 0.0, K_TEACHER, STUDENT_GRID, accs,
                              acc_min=0.90, wall_probe=probe, r_ib=r_ib)
    print(f"\n[4] preflight: {report['verdict']}")
    print(f"    items: {report['items']}")
    print(f"    wall geometry: {probe['wall']} (K* locatable: {probe['k_star_locatable']})")
    print("\nRead: R_IB/K_min are what the task NEEDS; K_wall is what a student of this family actually")
    print("needs; the certified exclusion is the only part that is a proof. See the README for what the")
    print("method cannot do (no sufficiency claims, no cross-task comparison, no prediction).")


if __name__ == "__main__":
    main()
