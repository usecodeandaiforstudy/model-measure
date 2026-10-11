"""WALL-SATURATION: why does the measured wall stop growing in the lattice family?

The observation to explain (SCALE-C round 3, fixed ceiling K_t = 2048, sigma = 1.0, 150 samples/class,
60 epochs, d = 64): the wall rose 8 -> 128 -> 512 for C = 8 -> 27 -> 64 and then STOPPED (C = 216 gave 512
again) while C = 1000 was not located at all (>2048 at a 26% gap). Three candidate mechanisms:

  M-BAYES  the task itself saturates: with spacing 3.0 and sigma 1.0 the class-conditional Gaussians
           overlap, so the Bayes accuracy is well below 1 and FALLS as C grows (more neighbours). A
           reference that already sits at the Bayes limit cannot be improved on, so "retain 90% of the
           reference" becomes an easy bar and the wall flattens. Signature: the reference's epsilon is at
           (or below) the Bayes error, and R_IB computed from the reference matches the Bayes R_IB.
  M-REF    the reference is weak for other reasons (budget/width), not because the task forbids better:
           a stronger reference raises the wall at fixed task and fixed ceiling.
  M-WIDTH  the 64-wide representation is the binding constraint: widening d raises the wall at fixed
           reference budget.
  M-KD     none of the above: the wall is a genuine property of distilling into a VQ student (an
           imitability ceiling of the soft-target objective).

This script answers M-BAYES with ZERO TRAINING (`--step0`): the lattice family is an equal-weight Gaussian
mixture with KNOWN centres, so both the Bayes error and the true mutual information I(X;Y) are computable
by Monte Carlo, and can be compared against the reference models the earlier rounds already trained.

Pre-registration (written 2026-10-05, before any number in this file was produced)
--------------------------------------------------------------------------------------------
(0) Ground truth by Monte Carlo (n_mc = 200k samples, seed 0), exact for the known mixture:
    - Bayes error eps_bayes(m, sigma): fraction where argmax_c log p(x|c) != y;
    - H(Y|X) = E[-log p(y|x)];  I(X;Y) = ln C - H(Y|X)   (classes uniform);
    - Bayes R_IB = ln C - h_b(eps_bayes) - eps_bayes*ln(C-1) (the Fano inversion of the BEST possible error).
(1) Cache audit: for every SCALE-C round-3 instance (m in {2,3,4,6,10}, sigma = 1.0), compare the trained
    reference's epsilon with eps_bayes(m, 1.0) and its R_IB with I(X;Y).
(2) Criteria for M-BAYES (the mechanism question this script is for):
    B1 the reference's epsilon is within +0.05 of eps_bayes at every C (i.e. it cannot be materially better);
    B2 the measured K_wall tracks the Bayes-implied floor: define K_min(bayes) = ceil(e^{R_IB(bayes)}); the
       wall should stop growing exactly when K_min(bayes) stops growing, i.e. the wall's shape follows the
       Bayes requirement rather than C;
    B3 the saturation ratio: K_wall / K_min(bayes) does not grow with C (the wall is a fixed multiple of
       the best possible requirement, not an independent scale).
    Verdict SUPPORT if B1 and (B2 or B3); MIXED if exactly one; REFUTED if B1 fails (then the task is NOT
    the binding constraint and the M-REF/M-WIDTH/M-KD arms are the ones to run).
(3) If M-BAYES is refuted, run the training arms in this order (each pre-registered the same way):
    M-REF:   sigma = 1.0, C = 216, reference = d 256 / 300 samples/class / 150 epochs vs the round-3
             baseline (d 64 / 150 / 60); criterion R1: the strong reference's eps drops by >= 0.05 AND the
             wall rises above 512.
    M-WIDTH: sigma = 1.0, C = 216, students at d = 256 under the same reference; criterion W1: the wall
             rises above 512 (i.e. the 64-wide representation, not K, was binding).
    M-KD:    if neither moves the wall, record the imitability ceiling as the surviving explanation.
(4) Relay: checkpoints under experiments/data/wall_sat/<fingerprint>/; step 0 is pure computation and
    caches nothing. Iron rules: criteria before data, English-only output, no downloads, no seed re-rolling.
--------------------------------------------------------------------------------------------
"""

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__)))
from capacity_lens import task_info_fano, DistillCapacity, measured_wall, looseness
from scale_c_demo import make_lattice, GridMLP, class_count
from scale_c_clean_demo import (K_GRID, K_TEACHER, SIGMA, EPOCHS, LR, N_PER_CLASS, SEED,
                                run_signature as clean_signature, train_one_epoch_cached, teacher_logits,
                                CKPT_ROOT as CLEAN_CKPT_ROOT, JSONL as CLEAN_JSONL)

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
JSONL = os.path.join(_DATA, "wall_sat.jsonl")
REPORT = os.path.join(_DATA, "wall_sat_report.txt")
SPACING = 3.0
N_MC = 200_000
M_GRID = [2, 3, 4, 6, 10]
SIGMA_GRID = [1.0, 0.7, 0.5, 0.35]


# ---------------------------------------------------------------------------
# Step 0: ground truth for the known mixture (no training, no model)
# ---------------------------------------------------------------------------

def lattice_centers(m, spacing=SPACING):
    coords = torch.cartesian_prod(*[torch.arange(m) for _ in range(3)]).float()
    return coords * spacing


@torch.no_grad()
def lattice_posterior_stats(m, sigma, n_mc=N_MC, spacing=SPACING, seed=0, chunk=20_000, device="cpu"):
    """Monte Carlo over the known mixture: Bayes error, H(Y|X), I(X;Y) and the Bayes-implied requirement.

    Exact in the limit: classes are uniform, centres known, likelihoods Gaussian with std sigma, so the
    Bayes decision rule is "nearest centre" and the posterior is a softmax over -||x-c||^2 / (2 sigma^2).
    """
    centers = lattice_centers(m, spacing).to(device)
    C = centers.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    err = 0.0
    h_cond = 0.0
    done = 0
    while done < n_mc:
        n = min(chunk, n_mc - done)
        y = torch.randint(0, C, (n,), generator=g)
        x = centers.cpu()[y] + sigma * torch.randn(n, 3, generator=g)
        x = x.to(device)
        d2 = torch.cdist(x, centers) ** 2
        logits = -d2 / (2.0 * sigma ** 2)
        log_post = torch.log_softmax(logits, dim=1)
        err += (log_post.argmax(dim=1).cpu() != y).sum().item()
        h_cond += -log_post[torch.arange(n), y].sum().item()
        done += n
    eps_bayes = err / n_mc
    h_yx = h_cond / n_mc
    h_y = math.log(C)
    mi = h_y - h_yx
    return dict(m=m, C=C, sigma=sigma, eps_bayes=eps_bayes, h_y=h_y, h_yx=h_yx, mi=mi,
                rib_bayes=task_info_fano(eps_bayes, C),
                k_min_bayes=DistillCapacity().min_codebook_size(task_info_fano(eps_bayes, C)),
                n_mc=n_mc)


# ---------------------------------------------------------------------------
# Cache audit: the round-3 references vs the Bayes ceiling
# ---------------------------------------------------------------------------

def load_clean_rows():
    """SCALE-C round-3 (clean design) rows from the recorded jsonl, keyed by C."""
    if not os.path.exists(CLEAN_JSONL):
        return {}
    rows = {}
    for line in open(CLEAN_JSONL, encoding="utf-8"):
        r = json.loads(line)
        rows[r["c"]] = r                       # later rows (the full run) overwrite the smoke run
    return rows


def step0(device, sigma_grid, m_grid):
    print("=" * 100)
    print("[WALL-SATURATION] step 0: the task's own ceiling (Bayes error and true I(X;Y)), zero training")
    print("=" * 100)
    print("  legend: eps_bayes = best achievable error; rib_bayes = Fano requirement of that error;")
    print("          k_min_bayes = ceil(e^rib_bayes); the 'reference' columns come from SCALE-C round 3\n")
    rows3 = load_clean_rows()
    print(f"  {'C':>5} {'m':>3} {'sigma':>6} {'eps_bayes':>10} {'I(X;Y)':>8} {'rib_bayes':>10} {'K_min_bayes':>12}"
          f" | {'ref eps':>8} {'ref R_IB':>9} {'K_min(ref)':>11} {'K_wall':>7}")
    out = []
    for sigma in sigma_grid:
        for m in m_grid:
            st = lattice_posterior_stats(m, sigma, device=device)
            r3 = rows3.get(st["C"], {})
            ref_eps = r3.get("eps")
            ref_rib = r3.get("rib")
            ref_kmin = r3.get("k_min")
            kwall = r3.get("k_wall")
            only_sigma1 = abs(sigma - 1.0) < 1e-9
            print(f"  {st['C']:>5} {m:>3} {sigma:>6.2f} {st['eps_bayes']:>10.3f} {st['mi']:>8.3f} "
                  f"{st['rib_bayes']:>10.3f} {st['k_min_bayes']:>12}"
                  + (f" | {ref_eps:>8.3f} {ref_rib:>9.3f} {ref_kmin:>11} {str(kwall):>7}"
                     if (only_sigma1 and ref_eps is not None) else " |      (no round-3 record)"))
            rec = dict(st)
            rec.update(step0=True, ref_eps=ref_eps, ref_rib=ref_rib, ref_k_min=ref_kmin, k_wall=kwall)
            out.append(rec)
    with open(JSONL, "a", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def bayes_verdict(rows, sigma=1.0, tol_eps=0.05):
    """B1: the reference is at the ceiling; B2/B3: the wall follows the Bayes requirement, not C."""
    detail = []
    audit = [r for r in rows if r.get("step0") and abs(r["sigma"] - sigma) < 1e-9 and r.get("ref_eps") is not None]
    if not audit:
        return "UNDECIDABLE", ["  no round-3 reference records to audit"]
    b1 = all(r["ref_eps"] <= r["eps_bayes"] + tol_eps for r in audit)
    detail.append(f"  B1 reference within +{tol_eps} of the Bayes error at every C: "
                  + ", ".join(f"C={r['C']}: {r['ref_eps']:.3f} vs {r['eps_bayes']:.3f}" for r in audit)
                  + f" -> {'PASS' if b1 else 'FAIL'}")
    located = [r for r in audit if r.get("k_wall")]
    if len(located) >= 2:
        ks = [r["k_min_bayes"] for r in located]
        ws = [r["k_wall"] for r in located]
        b2 = (ks[-1] - ks[0]) <= 2 and (ws[-1] - ws[0]) <= 2
        detail.append(f"  B2 Bayes floor and wall both flat over C={[r['C'] for r in located]}: "
                      f"K_min_bayes {ks} vs K_wall {ws} -> {'PASS' if b2 else 'FAIL'}")
        ratios = [r["k_wall"] / max(r["k_min_bayes"], 1) for r in located]
        b3 = ratios[-1] <= ratios[0] * 1.5
        detail.append(f"  B3 wall/Bayes-floor ratio not growing: {[round(x, 1) for x in ratios]} "
                      f"-> {'PASS' if b3 else 'FAIL'}")
    else:
        b2 = b3 = False
        detail.append("  B2/B3 not evaluable (fewer than 2 located walls)")
    if b1 and (b2 or b3):
        v = "SUPPORT: the wall flattens because the TASK flattens (M-BAYES)"
    elif b1 or b2 or b3:
        v = "MIXED: the reference sits at the ceiling but the wall does not follow it"
    else:
        v = "REFUTED: the task is not the binding constraint -> run the M-REF / M-WIDTH / M-KD arms"
    return v, detail


# ---------------------------------------------------------------------------
# The sigma arm: same C, same protocol, only the task's true information changes
# ---------------------------------------------------------------------------

def run_task_arm(m, configs, device, resume=True, epochs=EPOCHS, n_per_class=N_PER_CLASS, tag="arm"):
    """Train one reference + a student ladder for each (sigma, spacing) configuration of the lattice task.

    `configs` are (sigma, spacing) pairs; everything else is held fixed. Returns the recorded rows.
    Caching: experiments/data/wall_sat/m{m}_C{C}/<tag>_<sigma>_<spacing>/{teacher,targets,student_*}.pt
    """
    from torch.utils.data import DataLoader, TensorDataset
    from capacity_lens import train_one_epoch, evaluate

    C = class_count(m)
    ckpt_dir = os.path.join(_DATA, "wall_sat", f"m{m}_C{C}")
    os.makedirs(ckpt_dir, exist_ok=True)
    kt = K_TEACHER
    k_grid = sorted(set([K for K in K_GRID if K <= 2048] + [4096]))
    rows = []
    for sigma, spacing in configs:
        bayes = lattice_posterior_stats(m, sigma, spacing=spacing, device=device)
        t0 = time.time()
        (x_tr, y_tr), (x_te, y_te) = make_lattice(m, n_per_class=n_per_class, sigma=sigma,
                                                  spacing=spacing, seed=SEED)
        tr = DataLoader(TensorDataset(x_tr, y_tr), batch_size=128, shuffle=True, num_workers=0)
        te = DataLoader(TensorDataset(x_te, y_te), batch_size=512, shuffle=False, num_workers=0)
        stem = f"{tag}_s{sigma}_d{spacing}"
        path_t = os.path.join(ckpt_dir, f"teacher_{stem}.pt")
        if resume and os.path.exists(path_t):
            acc_t = float(torch.load(path_t, map_location="cpu", weights_only=True)["acc"])
        else:
            torch.manual_seed(70000 + C)
            ref = GridMLP(C, kt).to(device)
            opt = torch.optim.Adam(ref.parameters(), lr=LR)
            for _ in range(epochs):
                train_one_epoch(ref, tr, opt, device=device)
            acc_t, _ = evaluate(ref, te, device=device)
            torch.save({"acc": acc_t, "state_dict": ref.state_dict()}, path_t)
        ref = GridMLP(C, kt).to(device)
        ref.load_state_dict(torch.load(path_t, map_location=device, weights_only=True)["state_dict"])
        ref.eval()
        path_c = os.path.join(ckpt_dir, f"targets_{stem}.pt")
        if resume and os.path.exists(path_c):
            cache = torch.load(path_c, map_location="cpu", weights_only=True)
        else:
            cache = dict(tr=teacher_logits(ref, tr, device), te=teacher_logits(ref, te, device))
            torch.save(cache, path_c)
        tr_c = DataLoader(TensorDataset(x_tr, y_tr, cache["tr"].float()),
                          batch_size=128, shuffle=True, num_workers=0)
        accs = {}
        for K in k_grid:
            path_s = os.path.join(ckpt_dir, f"student_{stem}_K{K}.pt")
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
        eps = 1.0 - acc_t
        rib = task_info_fano(eps, C)
        k_min = DistillCapacity().min_codebook_size(rib)
        rel = {K: (acc_t - a) / max(acc_t, 1e-12) for K, a in accs.items()}
        k_wall, _ = measured_wall(sorted(accs), accs, acc_t, rel_tol=0.10)
        row = dict(arm=tag, c=C, m=m, sigma=sigma, spacing=spacing, n_per_class=n_per_class, epochs=epochs,
                   acc_t=acc_t, eps=eps, rib=rib, k_min=k_min, accs=accs, rel=rel, k_wall=k_wall,
                   looseness=looseness(rib, k_wall),
                   looseness_vs_bayes=looseness(bayes["rib_bayes"], k_wall),
                   eps_bayes=bayes["eps_bayes"], mi_true=bayes["mi"], rib_bayes=bayes["rib_bayes"],
                   k_min_bayes=bayes["k_min_bayes"], under_read_rib=bayes["rib_bayes"] - rib,
                   under_read_k=bayes["k_min_bayes"] / max(k_min, 1), t=time.time() - t0)
        with open(JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.append(row)
        print(f"  sigma={sigma:.2f} spacing={spacing:.1f}: ref acc={acc_t:.3f} (eps={eps:.3f} vs Bayes "
              f"{bayes['eps_bayes']:.3f}) R_IB={rib:.2f} (true I(X;Y)={bayes['mi']:.2f}) K_min={k_min} "
              f"(Bayes {bayes['k_min_bayes']}) -> K_wall={k_wall} "
              f"looseness={None if row['looseness'] is None else round(row['looseness'], 2)}  [{row['t']:.0f}s]")
        print("      rel gaps: " + "  ".join(f"K{K}:{rel[K]:.2f}" for K in sorted(accs)))
    return rows


def run_sigma_arm(m, sigmas, device, resume=True, epochs=EPOCHS, n_per_class=N_PER_CLASS):
    """At fixed C, vary sigma (the task's true information I(X;Y)) and re-measure the wall.

    Pre-registered criteria (recorded before the run, 2026-10-05):
      S1 the reference approaches Bayes where the task allows it: eps(ref) at the smallest sigma is <= 0.05,
         and eps(ref) - eps_bayes shrinks as sigma falls;
      S2 DOSE-RESPONSE: the wall rises with the task's true information, i.e. K_wall(sigma=0.5) > K_wall(1.0)
         = 512 at the same C (the wall is not a fixed protocol constant);
      S3 the wall stays within the historical looseness band of the BAYES requirement (K_wall/K_min(bayes)
         <= 50, the largest ratio seen in the 38-scenario survey) -- i.e. once the witness is strong, the
         wall follows the requirement instead of the protocol.
    Verdict: SUPPORT (reference-limited wall) if S2 and S3; MIXED if only S2; REFUTED (the flattening is a
    property of the distillation protocol itself, M-KD) if S2 fails.
    """
    print("=" * 100)
    print(f"[WALL-SATURATION] sigma arm at m={m} (C={class_count(m)}): same protocol, only the noise changes")
    print("=" * 100)
    print(f"  reference VQ(K_t={K_TEACHER}), students K in {sorted(set([K for K in K_GRID if K <= 2048] + [4096]))},"
          f" {epochs} ep, d=64, {n_per_class} samples/class\n")
    rows = run_task_arm(m, [(s, SPACING) for s in sigmas], device, resume=resume, epochs=epochs,
                        n_per_class=n_per_class, tag="sigma")
    print("\n  --- sigma-arm verdict ---")
    verdict, detail = sigma_verdict(rows)
    for line in detail:
        print(line)
    print(f"  [verdict] {verdict}")
    return rows, verdict


def sigma_verdict(rows):
    """S1-S3 on the sigma arm (see run_sigma_arm's docstring for the pre-registration)."""
    detail = []
    base = next((r for r in rows if abs(r["sigma"] - 1.0) < 1e-9), None)
    low = min(rows, key=lambda r: r["sigma"])
    s1 = low["eps"] <= 0.05 and (base is None or low["eps"] - low["eps_bayes"] < base["eps"] - base["eps_bayes"])
    detail.append(f"  S1 the witness approaches Bayes on an easier task: eps={low['eps']:.3f} at "
                  f"sigma={low['sigma']:.2f} (Bayes {low['eps_bayes']:.3f})"
                  + (f", gap {base['eps'] - base['eps_bayes']:.3f} -> {low['eps'] - low['eps_bayes']:.3f}"
                     if base else "") + f" -> {'PASS' if s1 else 'FAIL'}")
    if base is not None and base.get("k_wall") and low.get("k_wall"):
        s2 = low["k_wall"] > base["k_wall"]
        detail.append(f"  S2 dose-response: K_wall sigma={low['sigma']:.2f} -> {low['k_wall']} vs "
                      f"sigma=1.0 -> {base['k_wall']} -> {'PASS' if s2 else 'FAIL'}")
        s3 = low["looseness_vs_bayes"] is not None and low["looseness_vs_bayes"] <= 50.0
        detail.append(f"  S3 wall within 50x of the Bayes requirement: "
                      f"{None if low['looseness_vs_bayes'] is None else round(low['looseness_vs_bayes'], 1)}"
                      f" -> {'PASS' if s3 else 'FAIL'}")
    else:
        s2 = s3 = False
        detail.append("  S2/S3 not evaluable (no located wall at sigma=1.0 or at the smallest sigma)")
    if s2 and s3:
        v = "SUPPORT: the flattening was WITNESS-LIMITED (a stronger witness raises the wall)"
    elif s2:
        v = "MIXED: the wall responds to the task but stays far above the requirement"
    else:
        v = "REFUTED: the flattening is a property of the distillation protocol itself (M-KD)"
    return v, detail


def run_budget_probe(m, sigma, K, epoch_grid, device, resume=True, n_per_class=N_PER_CLASS):
    """Discriminate "imitability ceiling" from "optimization budget" at the hardest point of the sigma arm.

    At sigma=0.5 (C=216) the 60-epoch students plateau ~12% below the reference and adding codes does not
    help. If that plateau is an optimization artifact, giving the SAME student 3x the epochs must close the
    gap; if it is an intrinsic imitability ceiling of the distillation objective, it will not move.
    Pre-registered criterion P1: retention at 3x the budget improves by >= 0.05 (artifact) or by < 0.05
    (ceiling). Either answer is recorded as measured.
    """
    from torch.utils.data import DataLoader, TensorDataset
    from capacity_lens import train_one_epoch, evaluate

    C = class_count(m)
    ckpt_dir = os.path.join(_DATA, "wall_sat", f"m{m}_C{C}")
    os.makedirs(ckpt_dir, exist_ok=True)
    (x_tr, y_tr), (x_te, y_te) = make_lattice(m, n_per_class=n_per_class, sigma=sigma, spacing=SPACING, seed=SEED)
    te = DataLoader(TensorDataset(x_te, y_te), batch_size=512, shuffle=False, num_workers=0)
    path_t = os.path.join(ckpt_dir, f"teacher_sigma{sigma}.pt")
    if not os.path.exists(path_t):
        raise SystemExit(f"[WALL-SATURATION] the sigma-arm teacher for sigma={sigma} is required ({path_t})")
    acc_t = float(torch.load(path_t, map_location="cpu", weights_only=True)["acc"])
    ref = GridMLP(C, K_TEACHER).to(device)
    ref.load_state_dict(torch.load(path_t, map_location=device, weights_only=True)["state_dict"])
    ref.eval()
    tr = DataLoader(TensorDataset(x_tr, y_tr), batch_size=128, shuffle=True, num_workers=0)
    cache = dict(tr=teacher_logits(ref, tr, device), te=teacher_logits(ref, te, device))
    print("=" * 100)
    print(f"[WALL-SATURATION] KD budget probe: C={C}, sigma={sigma}, K={K}, epochs {epoch_grid}")
    print("=" * 100)
    print(f"  reference acc={acc_t:.3f}; the 60-epoch student's retention is in the sigma-arm jsonl\n")
    rows = []
    for ep in epoch_grid:
        path_s = os.path.join(ckpt_dir, f"student_budget{ep}_sigma{sigma}_K{K}.pt")
        if resume and os.path.exists(path_s):
            acc = float(torch.load(path_s, map_location="cpu", weights_only=True)["acc"])
        else:
            torch.manual_seed(71000 + C + K)
            stu = GridMLP(C, K).to(device)
            opt = torch.optim.Adam(stu.parameters(), lr=LR)
            loader = DataLoader(TensorDataset(x_tr, y_tr, cache["tr"].float()), batch_size=128,
                                shuffle=True, num_workers=0)
            for _ in range(ep):
                train_one_epoch_cached(stu, loader, opt, device=device)
            acc, _ = evaluate(stu, te, device=device)
            torch.save({"acc": acc, "state_dict": stu.state_dict()}, path_s)
        rel = (acc_t - acc) / acc_t
        rows.append(dict(arm="budget", c=C, m=m, sigma=sigma, k=K, epochs=ep, acc_t=acc_t, acc=acc, rel=rel))
        print(f"  epochs={ep:4d}: acc={acc:.3f} -> retention gap {rel:+.3f}")
    if len(rows) >= 2:
        gain = rows[0]["rel"] - rows[-1]["rel"]
        v = (f"REFUTED (budget artifact): 3x the epochs closes {gain:+.3f} of the gap"
             if gain >= 0.05 else
             f"SUPPORT (imitability ceiling): 3x the epochs changes the gap by only {gain:+.3f}")
        print(f"\n  P1 retention improved by {gain:+.3f} -> {v}")
    else:
        v = "UNDECIDABLE"
    for r in rows:
        with open(JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows, v


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def run_selftest():
    ok = []
    # a degenerate 1-class lattice has zero error and zero information
    st1 = lattice_posterior_stats(1, 1.0, n_mc=2000)
    ok.append((f"m=1 (single class): eps=0, I(X;Y)=0 -> {st1['eps_bayes']:.3f}, {st1['mi']:.3f}",
               abs(st1["eps_bayes"]) < 1e-9 and abs(st1["mi"]) < 1e-9))
    # error and information must move in opposite directions as the task gets noisier
    quiet = lattice_posterior_stats(3, 0.3, n_mc=20000)
    noisy = lattice_posterior_stats(3, 1.5, n_mc=20000)
    ok.append((f"noise raises Bayes error ({quiet['eps_bayes']:.3f} -> {noisy['eps_bayes']:.3f})",
               noisy["eps_bayes"] > quiet["eps_bayes"]))
    ok.append((f"noise lowers I(X;Y) ({quiet['mi']:.3f} -> {noisy['mi']:.3f})", noisy["mi"] < quiet["mi"]))
    # the Fano inversion clamps C to >= 2 (documented guard for the degenerate single-class case)
    ok.append((f"single-class case clamped by task_info_fano -> rib={st1['rib_bayes']:.3f}",
               abs(st1["rib_bayes"] - math.log(2)) < 1e-9))
    st = lattice_posterior_stats(4, 1.0, n_mc=20000)
    ok.append((f"Fano inversion consistent at m=4 (rib={st['rib_bayes']:.3f}, k_min={st['k_min_bayes']})",
               abs(st["rib_bayes"] - task_info_fano(st["eps_bayes"], 64)) < 1e-9
               and st["k_min_bayes"] == DistillCapacity().min_codebook_size(st["rib_bayes"])))
    print("[WALL-SATURATION selftest] ground-truth estimator + criteria:")
    all_ok = True
    for name, passed in ok:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        all_ok = all_ok and passed
    print(f"  [selftest verdict] {'all PASS' if all_ok else 'FAILURES present (code defect)'}")
    return all_ok


def verdict_only():
    rows = [json.loads(l) for l in open(JSONL, encoding="utf-8")] if os.path.exists(JSONL) else []
    if not rows:
        raise SystemExit("[WALL-SATURATION] nothing recorded yet")
    for sigma in sorted({r["sigma"] for r in rows if r.get("step0")}):
        v, detail = bayes_verdict(rows, sigma=sigma)
        print(f"[WALL-SATURATION offline re-verdict] sigma={sigma}")
        for line in detail:
            print(line)
        print(f"  [verdict] {v}")


def main():
    ap = argparse.ArgumentParser(description="WALL-SATURATION: mechanism of the flattening wall")
    ap.add_argument("--step0", action="store_true", help="ground-truth ceiling only (zero training)")
    ap.add_argument("--sigma-arm", action="store_true",
                    help="fixed C, vary sigma: is the flattening witness-limited or protocol-limited?")
    ap.add_argument("--budget-probe", action="store_true",
                    help="KD budget probe at the hardest sigma (imitability ceiling vs optimization budget)")
    ap.add_argument("--probe-sigma", type=float, default=0.5)
    ap.add_argument("--probe-k", type=int, default=4096)
    ap.add_argument("--probe-epochs", type=int, nargs="+", default=[60, 180])
    ap.add_argument("--m", type=int, default=6, help="lattice parameter for the sigma arm (m=6 -> C=216)")
    ap.add_argument("--sigma", type=float, nargs="+", default=[1.0, 0.7, 0.5])
    ap.add_argument("--fresh", action="store_true", help="recompute instead of relay mode")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verdict-only", action="store_true")
    ap.add_argument("--sigma-grid", type=float, nargs="+", default=SIGMA_GRID)
    ap.add_argument("--m-grid", type=int, nargs="+", default=M_GRID)
    args = ap.parse_args()
    if args.selftest:
        run_selftest()
        return
    if args.verdict_only:
        verdict_only()
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if args.budget_probe:
        rows, v = run_budget_probe(args.m, args.probe_sigma, args.probe_k, args.probe_epochs, device,
                                   resume=not args.fresh)
        with open(REPORT, "a" if os.path.exists(REPORT) else "w", encoding="utf-8") as f:
            f.write(f"budget probe sigma={args.probe_sigma} K={args.probe_k}: {v}\n")
        return
    if args.sigma_arm:
        rows, verdict = run_sigma_arm(args.m, args.sigma, device, resume=not args.fresh, epochs=args.epochs)
        with open(REPORT, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.write(verdict + "\n")
        print(f"  [report] {REPORT}")
        return
    rows = step0(device, args.sigma_grid, args.m_grid)
    print()
    v, detail = bayes_verdict(rows, sigma=1.0)
    for line in detail:
        print(line)
    print(f"  [verdict, sigma=1.0] {v}")
    with open(REPORT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write(v + "\n")
    print(f"  [report] {REPORT}")


if __name__ == "__main__":
    main()
