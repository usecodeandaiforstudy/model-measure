"""SCALE-AXES: the three scale axes that ARE reachable offline (sample count, parameter width, resolution).

Context. SCALE-C covered the CLASS-COUNT axis and WALL-SATURATION is covering the TASK-DIFFICULTY axis
(sigma). Three other axes are sometimes assumed to matter for "does the reading hold at scale". This
script measures the ones that can be reached without downloads:

  A1 SAMPLE COUNT   N per class in {40, 300} at C = 216, everything else fixed (the N = 150 point with the
                    full ladder comes from the WALL-SATURATION baseline). The task's true information
                    I(X;Y) does NOT depend on N (it is a property of the distribution), so any movement in
                    the reading is the WITNESS's quality showing through.
  A2 WIDTH          student representation width d in {64, 256} at fixed C, sigma and reference: is the
                    wall a function of the codebook alone, or does the width move it?
  A3 RESOLUTION     the project's own synthetic-image task (grid64: three bands x four positions, 64
                    classes) presented at 8x8, 16x16 and 32x32 -- generated at 16x16 with the frozen
                    encoding parameters and then average-pooled / nearest-upsampled, so the class
                    structure is identical and only the input resolution changes.

Pre-registration (2026-10-05, before any number below was produced)
--------------------------------------------------------------------------------------------
Ground truth: for A1/A2 the lattice family's I(X;Y) and Bayes error are known by Monte Carlo
(wall_saturation_demo.lattice_posterior_stats). For A3 there is no closed form; the claim tested is the
INVARIANCE of the reading under a change of input resolution at fixed class structure.

Criteria (each reported as PASS/FAIL with the number that decides it):
  A1-a  the reading approaches the truth as data grows: R_IB(600) > R_IB(40) and the under-reading
        (rib_bayes - rib) is smaller at N = 600;
  A1-b  the wall's looseness against the BAYES requirement falls with N. (The wall itself may move either
        way -- more data strengthens the reference AND the students -- so the criterion is stated against
        the fixed ground truth.)
        Verdict SUPPORT if A1-a and A1-b; MIXED if one; REFUTED if neither.
  A2    W1: the retention bar is crossed at a SMALLER K for d = 256 than for d = 64. If it does not move,
        the wall is a property of (K, task) and the width is irrelevant at this scale; both outcomes are
        recorded as measured.
  A3    R1: |R_IB(8x8) - R_IB(32x32)| <= 0.30 nats (the task's information is identical by construction);
        R2: the wall does not fall as resolution rises.
        Verdict SUPPORT if R1 and R2; MIXED if one; REFUTED if neither.

Scope note recorded in advance: "resolution" here is a manipulation of a synthetic-image task, not a
property of a natural dataset, and natural diversity remains unreachable offline.
Relay: checkpoints under experiments/data/scale_axes/<axis>_<config>/; --verdict-only re-reads the jsonl.
--------------------------------------------------------------------------------------------
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__)))
from capacity_lens import (VQBottleneck, task_info_fano, DistillCapacity, train_one_epoch, evaluate,
                           measured_wall, looseness)
from scale_c_demo import make_lattice, class_count
from scale_c_clean_demo import train_one_epoch_cached, teacher_logits
from wall_saturation_demo import lattice_posterior_stats, SPACING, SIGMA

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ROOT = os.path.join(_DATA, "scale_axes")
JSONL = os.path.join(_DATA, "scale_axes.jsonl")
REPORT = os.path.join(_DATA, "scale_axes_report.txt")

M_LATTICE = 6                       # C = 216
K_TEACHER = 2048
SAMPLE_GRID = [40, 300]             # N = 150 (full ladder) is measured by the WALL-SATURATION baseline
K_GRID_SAMPLES = [8, 32, 128, 512, 2048]
WIDTHS = [64, 128, 256]
K_GRID_WIDTH = [128, 256, 512, 2048]
RESOLUTIONS = [8, 16, 32]
K_GRID_RES = [8, 32, 128, 512]
EPOCHS_RES = 30
N_PER_CLASS_RES = 150
RES_HARD_EPOCHS = 6           # A3': protocol knobs that move the operating point off the ceiling
RES_HARD_KTEACHER = 64        #       (the generator's own parameters stay frozen: tuning them would be p-hacking)


class WideGridMLP(nn.Module):
    """GridMLP with a configurable representation width: Linear(3->d) -> ReLU -> VQ(d, K) -> Linear(d->C)."""

    def __init__(self, n_classes, k_codes, d=64):
        super().__init__()
        self.fc1 = nn.Linear(3, d)
        self.vq = VQBottleneck(d, k_codes, metric="euclid")
        self.fc2 = nn.Linear(d, n_classes)

    def forward(self, x):
        z, idx = self.vq(F.relu(self.fc1(x)))
        return self.fc2(z), [idx]


def _lattice_loaders(x_tr, y_tr, x_te, y_te, targets=None, batch=128):
    if targets is None:
        tr = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch, shuffle=True, num_workers=0)
    else:
        tr = DataLoader(TensorDataset(x_tr, y_tr, targets), batch_size=batch, shuffle=True, num_workers=0)
    te = DataLoader(TensorDataset(x_te, y_te), batch_size=512, shuffle=False, num_workers=0)
    return tr, te


def train_reference(cache, C, x_tr, y_tr, x_te, y_te, device, epochs, seed, d=64):
    """Train (or load) the reference and return (accuracy, cached train logits)."""
    path_t = os.path.join(cache, "teacher.pt")
    path_c = os.path.join(cache, "targets.pt")
    tr, te = _lattice_loaders(x_tr, y_tr, x_te, y_te)
    if os.path.exists(path_t):
        acc_t = float(torch.load(path_t, map_location="cpu", weights_only=True)["acc"])
    else:
        torch.manual_seed(seed)
        ref = WideGridMLP(C, K_TEACHER, d=d).to(device)
        opt = torch.optim.Adam(ref.parameters(), lr=1e-2)
        for _ in range(epochs):
            train_one_epoch(ref, tr, opt, device=device)
        acc_t, _ = evaluate(ref, te, device=device)
        torch.save({"acc": acc_t, "state_dict": ref.state_dict()}, path_t)
    ref = WideGridMLP(C, K_TEACHER, d=d).to(device)
    ref.load_state_dict(torch.load(path_t, map_location=device, weights_only=True)["state_dict"])
    ref.eval()
    if os.path.exists(path_c):
        targets = torch.load(path_c, map_location="cpu", weights_only=True)
    else:
        targets = teacher_logits(ref, tr, device)
        torch.save(targets, path_c)
    return acc_t, targets.float()


def train_student(cache, C, K, x_tr, y_tr, x_te, y_te, targets, device, epochs, seed, d=64, tag=None):
    """Train (or load) one VQ student under the cached-target distillation objective."""
    tag = tag or f"student_K{K}_d{d}"
    path = os.path.join(cache, f"{tag}.pt")
    if os.path.exists(path):
        d_ = torch.load(path, map_location="cpu", weights_only=True)
        return d_["acc"], d_["n_params"]
    torch.manual_seed(seed)
    model = WideGridMLP(C, K, d=d).to(device)
    tr, te = _lattice_loaders(x_tr, y_tr, x_te, y_te, targets=targets)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(epochs):
        train_one_epoch_cached(model, tr, opt, device=device)
    acc, _ = evaluate(model, te, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    torch.save({"acc": acc, "state_dict": model.state_dict(), "n_params": n_params}, path)
    return acc, n_params


def _record(row):
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# A1: sample count
# ---------------------------------------------------------------------------

def run_samples_axis(device, epochs, resume=True):
    C = class_count(M_LATTICE)
    bayes = lattice_posterior_stats(M_LATTICE, SIGMA, spacing=SPACING, device=device)
    rows = []
    print("=" * 100)
    print(f"[SCALE-AXES] A1 sample count: C={C}, sigma={SIGMA}, true I(X;Y)={bayes['mi']:.3f} nats "
          f"(independent of N by construction)")
    print("=" * 100)
    for n in SAMPLE_GRID:
        cache = os.path.join(ROOT, f"samples_N{n}")
        os.makedirs(cache, exist_ok=True)
        t0 = time.time()
        (x_tr, y_tr), (x_te, y_te) = make_lattice(M_LATTICE, n_per_class=n, sigma=SIGMA, spacing=SPACING, seed=0)
        acc_t, targets = train_reference(cache, C, x_tr, y_tr, x_te, y_te, device, epochs, 70000 + C)
        accs = {}
        for K in K_GRID_SAMPLES:
            accs[K], _ = train_student(cache, C, K, x_tr, y_tr, x_te, y_te, targets, device, epochs,
                                       71000 + C + K)
        eps = 1.0 - acc_t
        rib = task_info_fano(eps, C)
        rel = {K: (acc_t - a) / max(acc_t, 1e-12) for K, a in accs.items()}
        k_wall, _ = measured_wall(sorted(accs), accs, acc_t, rel_tol=0.10)
        row = dict(axis="samples", n_per_class=n, c=C, sigma=SIGMA, epochs=epochs, acc_t=acc_t, eps=eps,
                   rib=rib, k_min=DistillCapacity().min_codebook_size(rib), accs=accs, rel=rel,
                   k_wall=k_wall, looseness=looseness(rib, k_wall),
                   looseness_vs_bayes=looseness(bayes["rib_bayes"], k_wall), rib_bayes=bayes["rib_bayes"],
                   mi_true=bayes["mi"], eps_bayes=bayes["eps_bayes"],
                   under_read_rib=bayes["rib_bayes"] - rib, t=time.time() - t0)
        _record(row)
        rows.append(row)
        print(f"  N={n:4d}/class: ref acc={acc_t:.3f} (eps={eps:.3f}, Bayes {bayes['eps_bayes']:.3f}) "
              f"R_IB={rib:.2f} (true {bayes['mi']:.2f}) K_min={row['k_min']} -> K_wall={k_wall} "
              f"looseness_vs_Bayes="
              f"{None if row['looseness_vs_bayes'] is None else round(row['looseness_vs_bayes'], 1)}"
              f"  [{row['t']:.0f}s]")
        print("      rel gaps: " + "  ".join(f"K{K}:{rel[K]:.2f}" for K in sorted(accs)))
    return rows


def samples_verdict(rows):
    detail = []
    lo = min(rows, key=lambda r: r["n_per_class"])
    hi = max(rows, key=lambda r: r["n_per_class"])
    a = hi["rib"] > lo["rib"] and hi["under_read_rib"] < lo["under_read_rib"]
    detail.append(f"  A1-a the reading approaches the truth with more data: R_IB {lo['rib']:.3f} "
                  f"(N={lo['n_per_class']}) -> {hi['rib']:.3f} (N={hi['n_per_class']}), under-reading "
                  f"{lo['under_read_rib']:.3f} -> {hi['under_read_rib']:.3f} -> {'PASS' if a else 'FAIL'}")
    if lo.get("looseness_vs_bayes") is not None and hi.get("looseness_vs_bayes") is not None:
        b = hi["looseness_vs_bayes"] < lo["looseness_vs_bayes"]
        detail.append(f"  A1-b looseness against the Bayes requirement falls: {lo['looseness_vs_bayes']:.1f} "
                      f"-> {hi['looseness_vs_bayes']:.1f} -> {'PASS' if b else 'FAIL'}")
    else:
        b = False
        detail.append("  A1-b not evaluable (a wall was not located at one end)")
    if a and b:
        return "SUPPORT: the reading is data-limited, not task-limited", detail
    if a or b:
        return "MIXED: only one of the two data-sufficiency criteria holds", detail
    return "REFUTED: more data does not tighten the reading", detail


# ---------------------------------------------------------------------------
# A2: width
# ---------------------------------------------------------------------------

def run_width_axis(device, epochs, resume=True):
    C = class_count(M_LATTICE)
    rows = []
    print("=" * 100)
    print(f"[SCALE-AXES] A2 width: C={C}, sigma={SIGMA}, K in {K_GRID_WIDTH}, student d in {WIDTHS}")
    print("=" * 100)
    (x_tr, y_tr), (x_te, y_te) = make_lattice(M_LATTICE, n_per_class=150, sigma=SIGMA, spacing=SPACING, seed=0)
    base_cache = os.path.join(ROOT, "width_base")
    os.makedirs(base_cache, exist_ok=True)
    acc_t, targets = train_reference(base_cache, C, x_tr, y_tr, x_te, y_te, device, epochs, 70000 + C)
    print(f"  reference: acc={acc_t:.3f} (d=64, K_t={K_TEACHER}) -- shared by every width below\n")
    for d in WIDTHS:
        cache = os.path.join(ROOT, f"width_d{d}")
        os.makedirs(cache, exist_ok=True)
        accs, params = {}, {}
        for K in K_GRID_WIDTH:
            accs[K], params[K] = train_student(cache, C, K, x_tr, y_tr, x_te, y_te, targets, device, epochs,
                                               71000 + C + K + d, d=d)
        rel = {K: (acc_t - a) / max(acc_t, 1e-12) for K, a in accs.items()}
        k_wall, _ = measured_wall(sorted(accs), accs, acc_t, rel_tol=0.10)
        row = dict(axis="width", d=d, c=C, sigma=SIGMA, epochs=epochs, acc_t=acc_t, accs=accs, rel=rel,
                   k_wall=k_wall, params=params, k_grid=K_GRID_WIDTH)
        _record(row)
        rows.append(row)
        print(f"  d={d:3d}: accs={ {K: round(a, 3) for K, a in accs.items()} } "
              f"params={ {K: params[K] for K in sorted(params)} } -> K_wall={k_wall}")
        print("      rel gaps: " + "  ".join(f"K{K}:{rel[K]:.2f}" for K in sorted(accs)))
    return rows


def width_verdict(rows):
    """W1 as pre-registered (does the located wall move?), plus the robustness reading that decides how much
    the located wall is worth: the wall is a threshold on a FLAT part of the gap curve, so the located value
    is stable only if the retention differences at fixed K are larger than the curve's own slack.

    Amendment recorded after the first pass (2026-10-05): the first run gave K_wall = 2048 for d=64 and 512
    for d=256, but the underlying retention at K=512 differed by only 2 percentage points (0.902 vs 0.918),
    which is boundary noise rather than an effect. A third width (d=128) and a K=256 rung were added to test
    monotonicity; W1 keeps its pre-registered wording, and W2 is the robustness criterion that was missing.
    """
    detail = []
    by_d = {r["d"]: r for r in rows}
    if 64 in by_d and 256 in by_d and by_d[64]["k_wall"] and by_d[256]["k_wall"]:
        k64, k256 = by_d[64]["k_wall"], by_d[256]["k_wall"]
        moved = k256 < k64
        detail.append(f"  W1 (pre-registered) the located wall moves with the student's width: "
                      f"d=64 -> K_wall={k64}, d=256 -> K_wall={k256} -> "
                      f"{'PASS' if moved else 'FAIL'}")
        for K in sorted(set(by_d[64]["rel"]) & set(by_d[256]["rel"])):
            r64, r256 = by_d[64]["rel"][K], by_d[256]["rel"][K]
            detail.append(f"       retention gap at K={K}: d=64 {r64:+.3f} vs d=256 {r256:+.3f} "
                          f"(difference {abs(r64 - r256):.3f})")
        # W2 (added): the width effect must show up as a retention gain larger than the curve's slack
        common = sorted(set(by_d[64]["rel"]) & set(by_d[256]["rel"]))
        gains = [by_d[64]["rel"][K] - by_d[256]["rel"][K] for K in common]
        best = max(gains) if gains else 0.0
        w2 = best >= 0.03
        detail.append(f"  W2 (added for robustness) the largest retention gain from widening is >= 0.03: "
                      f"{best:+.3f} -> {'PASS' if w2 else 'FAIL (the located-wall move was boundary noise)'}")
        if 128 in by_d:
            monotone = all(by_d[128]["rel"][K] >= min(by_d[64]["rel"][K], by_d[256]["rel"][K]) - 0.02
                           for K in common)
            detail.append(f"  W3 (added) d=128 sits between the two in retention: "
                          f"{ {K: round(by_d[128]['rel'][K], 3) for K in common} } -> "
                          f"{'consistent' if monotone else 'NOT monotone'}")
        if moved and w2:
            return ("SUPPORT (marginal): a wider student retains the reference at a smaller K, and the "
                    "largest retention gain clears the added threshold -- but only just (see the numbers "
                    "above), so the effect is suggestive rather than established"), detail
        if moved:
            return ("MIXED/REFUTED: the located wall moves, but the underlying retention difference is within "
                    "the criterion's own boundary slack -- the wall is not stable enough at this scale to "
                    "attribute the move to width"), detail
        return ("MIXED/REFUTED: the width does not move the wall at this scale (K is the binding quantity)",
                detail)
    detail.append("  W1 not evaluable (a wall was not located for both widths)")
    return "UNDECIDABLE", detail


# ---------------------------------------------------------------------------
# A3: resolution
# ---------------------------------------------------------------------------

def make_grid64_at(resolution, n_per_class=N_PER_CLASS_RES, seed=0):
    """The frozen grid64 task (cifar_demo.make_grid64_images, 16x16; encoding parameters NOT tuned) presented
    at another resolution: average-pool 2x2 for 8x8, nearest-neighbour 2x up for 32x32."""
    from cifar_demo import make_grid64_images
    (x_tr, y_tr), (x_te, y_te) = make_grid64_images(n_per_class=n_per_class, seed=seed)

    def resample(x):
        if resolution == 16:
            return x
        if resolution == 8:
            return F.avg_pool2d(x, 2)
        if resolution == 32:
            return F.interpolate(x, scale_factor=2.0, mode="nearest")
        raise ValueError(f"unsupported resolution {resolution}")

    return (resample(x_tr), y_tr), (resample(x_te), y_te)


class PixelMLP(nn.Module):
    """One architecture at every resolution: Linear(r^2 -> 64) -> ReLU -> VQ(64, K) -> Linear(64 -> C)."""

    def __init__(self, resolution, n_classes, k_codes, d=64):
        super().__init__()
        self.fc1 = nn.Linear(resolution * resolution, d)
        self.vq = VQBottleneck(d, k_codes, metric="euclid")
        self.fc2 = nn.Linear(d, n_classes)

    def forward(self, x):
        z, idx = self.vq(F.relu(self.fc1(x.flatten(1))))
        return self.fc2(z), [idx]


def run_resolution_axis(device, epochs=EPOCHS_RES, resume=True):
    C = 64
    rows = []
    print("=" * 100)
    print(f"[SCALE-AXES] A3 resolution: the grid64 image task at {RESOLUTIONS} "
          f"(generated at 16x16, pooled/upsampled; one MLP architecture throughout)")
    print("=" * 100)
    for res in RESOLUTIONS:
        cache = os.path.join(ROOT, f"res_{res}")
        os.makedirs(cache, exist_ok=True)
        t0 = time.time()
        (x_tr, y_tr), (x_te, y_te) = make_grid64_at(res)
        xtr, xte = x_tr.flatten(1), x_te.flatten(1)
        tr = DataLoader(TensorDataset(xtr, y_tr), batch_size=128, shuffle=True, num_workers=0)
        te = DataLoader(TensorDataset(xte, y_te), batch_size=512, shuffle=False, num_workers=0)
        path_t = os.path.join(cache, "teacher.pt")
        if resume and os.path.exists(path_t):
            acc_t = float(torch.load(path_t, map_location="cpu", weights_only=True)["acc"])
        else:
            torch.manual_seed(70000 + C + res)
            ref = PixelMLP(res, C, K_TEACHER).to(device)
            opt = torch.optim.Adam(ref.parameters(), lr=1e-3)
            for _ in range(epochs):
                train_one_epoch(ref, tr, opt, device=device)
            acc_t, _ = evaluate(ref, te, device=device)
            torch.save({"acc": acc_t, "state_dict": ref.state_dict()}, path_t)
        ref = PixelMLP(res, C, K_TEACHER).to(device)
        ref.load_state_dict(torch.load(path_t, map_location=device, weights_only=True)["state_dict"])
        ref.eval()
        path_c = os.path.join(cache, "targets.pt")
        if resume and os.path.exists(path_c):
            targets = torch.load(path_c, map_location="cpu", weights_only=True)
        else:
            targets = teacher_logits(ref, tr, device)
            torch.save(targets, path_c)
        targets = targets.float()
        accs = {}
        for K in K_GRID_RES:
            path_s = os.path.join(cache, f"student_K{K}.pt")
            if resume and os.path.exists(path_s):
                accs[K] = float(torch.load(path_s, map_location="cpu", weights_only=True)["acc"])
                continue
            torch.manual_seed(71000 + C + K + res)
            stu = PixelMLP(res, C, K).to(device)
            opt = torch.optim.Adam(stu.parameters(), lr=1e-3)
            loader = DataLoader(TensorDataset(xtr, y_tr, targets), batch_size=128, shuffle=True, num_workers=0)
            for _ in range(epochs):
                train_one_epoch_cached(stu, loader, opt, device=device)
            acc, _ = evaluate(stu, te, device=device)
            torch.save({"acc": acc, "state_dict": stu.state_dict()}, path_s)
            accs[K] = acc
        eps = 1.0 - acc_t
        rib = task_info_fano(eps, C)
        rel = {K: (acc_t - a) / max(acc_t, 1e-12) for K, a in accs.items()}
        k_wall, _ = measured_wall(sorted(accs), accs, acc_t, rel_tol=0.10)
        row = dict(axis="resolution", resolution=res, c=C, epochs=epochs, acc_t=acc_t, eps=eps, rib=rib,
                   k_min=DistillCapacity().min_codebook_size(rib), accs=accs, rel=rel, k_wall=k_wall,
                   looseness=looseness(rib, k_wall), t=time.time() - t0)
        _record(row)
        rows.append(row)
        print(f"  {res}x{res}: ref acc={acc_t:.3f} (eps={eps:.3f}) R_IB={rib:.2f} K_min={row['k_min']} "
              f"-> K_wall={k_wall}  [{row['t']:.0f}s]")
        print("      rel gaps: " + "  ".join(f"K{K}:{rel[K]:.2f}" for K in sorted(accs)))
    return rows


def resolution_hard_verdict(rows):
    detail = []
    by_res = {r["resolution"]: r for r in rows}
    ceiling = math.log(64)
    r3 = all(r["rib"] < ceiling - 0.05 for r in rows)
    detail.append("  R3' non-degenerate at every resolution: "
                  + ", ".join(f"{r['resolution']}x{r['resolution']} R_IB={r['rib']:.3f}" for r in rows)
                  + f" (ceiling {ceiling:.3f}) -> {'PASS' if r3 else 'FAIL'}")
    if 8 in by_res and 32 in by_res:
        diff = abs(by_res[8]["rib"] - by_res[32]["rib"])
        r1 = diff <= 0.30
        detail.append(f"  R1' the reading is resolution-invariant: |diff|={diff:.3f} (tol 0.30) -> "
                      f"{'PASS' if r1 else 'FAIL'}")
        walls = {r["resolution"]: r["k_wall"] for r in rows}
        r2 = all(v is not None for v in walls.values()) and walls[32] >= walls[8]
        detail.append(f"  R2' the wall does not fall as the resolution rises: {walls} -> "
                      f"{'PASS' if r2 else 'FAIL'}")
        if r1 and r2 and r3:
            return "SUPPORT: the resolution invariance holds at a non-degenerate operating point too", detail
        if sum([r1, r2, r3]) >= 2:
            return "MIXED: two of the three criteria hold", detail
        return "REFUTED: the resolution collapse reappears off the saturated point", detail
    detail.append("  R1' not evaluable (8x8 or 32x32 missing)")
    return "UNDECIDABLE", detail


def run_resolution_hard(device, epochs=RES_HARD_EPOCHS, k_teacher=RES_HARD_KTEACHER, resume=True):
    """A3': the resolution axis at a NON-DEGENERATE operating point.

    The first A3 run reported acc = 1.000 at every resolution, which puts R_IB at its ln C ceiling and K_min at
    C = 64 -- the invariance it demonstrated was therefore trivial. The generator's encoding parameters are
    frozen by their own docstring (tuning them to suit the result would be p-hacking), so the operating point is
    moved with PROTOCOL knobs instead: a small reference codebook and a short budget, both declared in advance.
    Criteria R1'/R2'/R3' are pre-registered in resolution_hard_verdict.
    """
    C = 64
    rows = []
    print("=" * 100)
    print(f"[SCALE-AXES] A3' resolution at a NON-DEGENERATE point: reference VQ({k_teacher}), {epochs} epochs, "
          f"resolutions {RESOLUTIONS}")
    print("=" * 100)
    for res in RESOLUTIONS:
        cache = os.path.join(ROOT, f"res_hard_{res}")
        os.makedirs(cache, exist_ok=True)
        t0 = time.time()
        (x_tr, y_tr), (x_te, y_te) = make_grid64_at(res)
        xtr, xte = x_tr.flatten(1), x_te.flatten(1)
        tr = DataLoader(TensorDataset(xtr, y_tr), batch_size=128, shuffle=True, num_workers=0)
        te = DataLoader(TensorDataset(xte, y_te), batch_size=512, shuffle=False, num_workers=0)
        path_t = os.path.join(cache, "teacher.pt")
        if resume and os.path.exists(path_t):
            acc_t = float(torch.load(path_t, map_location="cpu", weights_only=True)["acc"])
        else:
            torch.manual_seed(70000 + C + res)
            ref = PixelMLP(res, C, k_teacher).to(device)
            opt = torch.optim.Adam(ref.parameters(), lr=1e-3)
            for _ in range(epochs):
                train_one_epoch(ref, tr, opt, device=device)
            acc_t, _ = evaluate(ref, te, device=device)
            torch.save({"acc": acc_t, "state_dict": ref.state_dict()}, path_t)
        ref = PixelMLP(res, C, k_teacher).to(device)
        ref.load_state_dict(torch.load(path_t, map_location=device, weights_only=True)["state_dict"])
        ref.eval()
        path_c = os.path.join(cache, "targets.pt")
        if resume and os.path.exists(path_c):
            targets = torch.load(path_c, map_location="cpu", weights_only=True).float()
        else:
            with torch.no_grad():
                targets = torch.cat([ref(x.to(device))[0].half().cpu()
                                     for x, _ in DataLoader(TensorDataset(xtr, y_tr), batch_size=512)]).float()
            torch.save(targets, path_c)
        accs = {}
        for K in K_GRID_RES:
            path_s = os.path.join(cache, f"student_K{K}.pt")
            if resume and os.path.exists(path_s):
                accs[K] = float(torch.load(path_s, map_location="cpu", weights_only=True)["acc"])
                continue
            torch.manual_seed(71000 + C + K + res)
            stu = PixelMLP(res, C, K).to(device)
            opt = torch.optim.Adam(stu.parameters(), lr=1e-3)
            loader = DataLoader(TensorDataset(xtr, y_tr, targets), batch_size=128, shuffle=True, num_workers=0)
            for _ in range(epochs):
                train_one_epoch_cached(stu, loader, opt, device=device)
            acc, _ = evaluate(stu, te, device=device)
            torch.save({"acc": acc, "state_dict": stu.state_dict()}, path_s)
            accs[K] = acc
        eps = 1.0 - acc_t
        rib = task_info_fano(eps, C)
        rel = {K: (acc_t - a) / max(acc_t, 1e-12) for K, a in accs.items()}
        k_wall, _ = measured_wall(sorted(accs), accs, acc_t, rel_tol=0.10)
        row = dict(axis="resolution_hard", resolution=res, c=C, epochs=epochs, k_teacher=k_teacher,
                   acc_t=acc_t, eps=eps, rib=rib, k_min=DistillCapacity().min_codebook_size(rib), accs=accs,
                   rel=rel, k_wall=k_wall, looseness=looseness(rib, k_wall), t=time.time() - t0)
        with open(JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.append(row)
        print(f"  {res}x{res}: ref acc={acc_t:.3f} (eps={eps:.3f}) R_IB={rib:.3f} (ceiling {math.log(C):.3f}) "
              f"K_min={row['k_min']} -> K_wall={k_wall}  [{row['t']:.0f}s]")
        print("      rel gaps: " + "  ".join(f"K{K}:{rel[K]:.2f}" for K in sorted(accs)))
    return rows


def resolution_verdict(rows):
    detail = []
    by_res = {r["resolution"]: r for r in rows}
    if 8 in by_res and 32 in by_res:
        diff = abs(by_res[8]["rib"] - by_res[32]["rib"])
        r1 = diff <= 0.30
        detail.append(f"  R1 the reading is resolution-invariant: R_IB(8x8)={by_res[8]['rib']:.3f} vs "
                      f"R_IB(32x32)={by_res[32]['rib']:.3f}, |diff|={diff:.3f} (tol 0.30) -> "
                      f"{'PASS' if r1 else 'FAIL'}")
        walls = {r["resolution"]: r["k_wall"] for r in rows}
        r2 = all(v is not None for v in walls.values()) and walls[32] >= walls[8]
        detail.append(f"  R2 the wall does not fall as the resolution rises: {walls} -> "
                      f"{'PASS' if r2 else 'FAIL'}")
        if r1 and r2:
            return "SUPPORT: the reading is a property of the task, not of the input resolution", detail
        if r1 or r2:
            return "MIXED: the reading survives the resolution change but the wall moves", detail
        return "REFUTED: the reading depends on the input resolution", detail
    detail.append("  R1 not evaluable (8x8 or 32x32 missing)")
    return "UNDECIDABLE", detail


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_selftest():
    ok = []
    (a_tr, _), (a_te, _) = make_grid64_at(8)
    (c_tr, _), (c_te, _) = make_grid64_at(32)
    ok.append((f"resampling keeps the class structure (8x8 {tuple(a_tr.shape)}, 32x32 {tuple(c_tr.shape)})",
               a_tr.shape[-1] == 8 and c_tr.shape[-1] == 32 and a_tr.shape[0] == c_tr.shape[0]))
    b_tr, _ = make_grid64_at(16)[0]
    ok.append(("the 16x16 path returns the frozen generator unchanged", tuple(b_tr.shape[1:]) == (1, 16, 16)))
    ok.append(("pooling preserves the mean intensity (no new information at 8x8)",
               torch.allclose(F.avg_pool2d(b_tr[:8], 2).mean(), a_tr[:8].mean(), atol=1e-6)))
    ok.append(("upsampling preserves the mean intensity (no new information at 32x32)",
               torch.allclose(c_tr[:8].mean(), b_tr[:8].mean(), atol=1e-6)))
    v, _ = resolution_verdict([dict(resolution=8, rib=1.0, k_wall=8), dict(resolution=16, rib=1.1, k_wall=16),
                               dict(resolution=32, rib=1.2, k_wall=32)])
    ok.append(("resolution criteria accept an invariant reading", v.startswith("SUPPORT")))
    v2, _ = samples_verdict([dict(n_per_class=40, rib=1.0, under_read_rib=2.0, looseness_vs_bayes=9.0),
                             dict(n_per_class=600, rib=2.0, under_read_rib=1.0, looseness_vs_bayes=4.0)])
    ok.append(("sample-count criteria accept a data-limited reading", v2.startswith("SUPPORT")))
    print("[SCALE-AXES selftest] axis criteria + the resolution manipulation:")
    all_ok = True
    for name, passed in ok:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        all_ok = all_ok and passed
    print(f"  [selftest verdict] {'all PASS' if all_ok else 'FAILURES present (code defect)'}")
    return all_ok


def verdict_only():
    rows = [json.loads(l) for l in open(JSONL, encoding="utf-8")] if os.path.exists(JSONL) else []
    if not rows:
        raise SystemExit("[SCALE-AXES] nothing recorded yet")
    for axis, fn in (("samples", samples_verdict), ("width", width_verdict), ("resolution", resolution_verdict)):
        sub = [r for r in rows if r.get("axis") == axis]
        if not sub:
            continue
        v, detail = fn(sub)
        print(f"[SCALE-AXES offline re-verdict] {axis}")
        for line in detail:
            print(line)
        print(f"  [verdict] {v}")


def main():
    ap = argparse.ArgumentParser(description="SCALE-AXES: sample count, width, resolution")
    ap.add_argument("--axis", choices=["samples", "width", "resolution", "resolution-hard", "all"],
                    default="all")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--res-epochs", type=int, default=EPOCHS_RES)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verdict-only", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        run_selftest()
        return
    if args.verdict_only:
        verdict_only()
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    resume = not args.fresh
    summary = []
    if args.axis in ("samples", "all"):
        rows = run_samples_axis(device, args.epochs, resume=resume)
        v, detail = samples_verdict(rows)
        print()
        for line in detail:
            print(line)
        print(f"  [verdict A1 sample count] {v}")
        summary.append(("A1 sample count", v))
    if args.axis in ("width", "all"):
        rows = run_width_axis(device, args.epochs, resume=resume)
        v, detail = width_verdict(rows)
        print()
        for line in detail:
            print(line)
        print(f"  [verdict A2 width] {v}")
        summary.append(("A2 width", v))
    if args.axis in ("resolution", "resolution-hard", "all"):
        if args.axis in ("resolution", "all"):
            rows = run_resolution_axis(device, epochs=args.res_epochs, resume=resume)
            v, detail = resolution_verdict(rows)
            print()
            for line in detail:
                print(line)
            print(f"  [verdict A3 resolution] {v}")
            summary.append(("A3 resolution", v))
        if args.axis in ("resolution-hard", "all"):
            rows = run_resolution_hard(device, resume=resume)
            v, detail = resolution_hard_verdict(rows)
            print()
            for line in detail:
                print(line)
            print(f"  [verdict A3-prime resolution at a non-degenerate point] {v}")
            summary.append(("A3-prime resolution (non-degenerate)", v))
    with open(REPORT, "w", encoding="utf-8") as f:
        for name, v in summary:
            f.write(f"{name}: {v}\n")
    print("\n  [summary] " + " | ".join(f"{n}: {v.split(':')[0]}" for n, v in summary))
    print(f"  [report] {REPORT}")


if __name__ == "__main__":
    main()
