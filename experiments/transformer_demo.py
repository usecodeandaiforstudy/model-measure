"""Capacity Lens validation experiment · Transformer line: T1 tightness / T2 failure of min under the residual stream.

**Why this line is needed**: MNIST+MLP and CIFAR+CNN are both "feed-forward" structures, whereas a Transformer block is
**residual + parallel** (multi-head attention || FFN, each with its own identity bypass) -- exactly the kind of topology for which P4 determined
that the `min` bound fails. So the Transformer is not just "one more architecture"; it is the real carrier of the P4 conclusion:

  T1: whether the log K upper bound is likewise **tight** on Transformer representations (whether codebook utilization is healthy);
  T2: after inserting a narrow VQ bottleneck into the FFN branch, **whether the residual stream makes the min lock-in effect disappear** (P4 on a real architecture).

Usage:
    python experiments/transformer_demo.py              # all experiments (slower on CPU)
    python experiments/transformer_demo.py --fast       # reduced budget, **still outputs a verdict**
    python experiments/transformer_demo.py --quick      # smoke test, no verdict output
    python experiments/transformer_demo.py --only t2    # run a single experiment

The success/failure criteria of each experiment are written in the respective verdict function.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# let the script import the parent package and the data utilities in the same directory
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, ".."))
sys.path.append(_HERE)
from capacity_lens import (VQBottleneck, train_one_epoch, evaluate, entropy_of,
                           discrete_capacity_logK, min_capacity_serial,
                           sum_capacity_parallel)
from mnist_demo import get_mnist, count_params


# ---------------------------------------------------------------------------
# Model: a small GPT-2-style decoder-only Transformer
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """pre-LN residual block: x = x + attn(ln1(x)); x = x + ffn(ln2(x)).

    When residual=False both residual connections are removed (x = attn(...); x = ffn(...)) and the block degenerates into
    **pure serial** -- this is the control arm of T2, used to test the single independent variable "with/without bypass".
    A VQ bottleneck (k_ffn) may optionally be inserted inside the FFN branch; with residuals it is bypassed by identity,
    and without residuals it enters the serial path.

    Metric of the bottleneck: after GELU the FFN hidden layer is **non-negative** and its dimensions have uneven scales;
    cosine nearest-neighbor would squeeze all vectors into the same direction of the positive quadrant (poor discriminability),
    so Euclidean nearest-neighbor is used here (metric="euclid"); T1's pooled representation comes after LayerNorm
    (approximately zero-mean, unit-variance) and keeps the default cosine.
    """

    def __init__(self, d_model, n_head, d_ff, k_ffn=None, residual=True):
        super().__init__()
        self.residual = residual
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.vq = VQBottleneck(d_ff, k_ffn, metric="euclid") if k_ffn else None
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x, mask):
        n = self.ln1(x)
        a, _ = self.attn(n, n, n, attn_mask=mask, need_weights=False)
        x = x + a if self.residual else a
        h = F.gelu(self.fc1(self.ln2(x)))
        idx = None
        if self.vq is not None:
            h, idx = self.vq(h)
        f = self.fc2(h)
        x = x + f if self.residual else f
        return x, idx


class TinyGPT(nn.Module):
    """Sequence classifier that cuts MNIST into patches as a "sequence" (causal attention, classifying from the pooled representation).

    - Input 28×28 -> (28/patch)² patches, each patch linearly projected to d_model;
    - n_layer pre-LN causal attention blocks (residuals can be on/off);
    - LayerNorm -> sequence mean pooling -> optional **representation bottleneck** (used by T1) -> Linear(→10).

    About 60k parameters (d_model=48, n_layer=2), runnable on CPU; smaller than the CIFAR CNN,
    but structurally a real Transformer (multi-head, causal mask, pre-LN residuals, GELU FFN).
    """

    def __init__(self, d_model=48, n_head=3, n_layer=2, patch=7, ffn_mult=4,
                 k_ffn=None, k_pool=None, residual=True):
        super().__init__()
        n_side = 28 // patch
        if n_side * patch != 28:
            raise ValueError(f"patch={patch} does not divide 28")
        self.patch = patch
        self.n_tokens = n_side * n_side
        self.proj = nn.Linear(patch * patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        self.blocks = nn.ModuleList([
            Block(d_model, n_head, d_model * ffn_mult, k_ffn, residual)
            for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.vq_pool = VQBottleneck(d_model, k_pool) if k_pool else None
        self.head = nn.Linear(d_model, 10)

    def forward(self, x):
        b = x.size(0)
        patches = F.unfold(x.view(b, 1, 28, 28), kernel_size=self.patch,
                           stride=self.patch)                 # [B, p*p, n_tokens]
        h = self.proj(patches.transpose(1, 2)) + self.pos      # [B, n_tokens, d_model]
        length = h.size(1)
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool,
                                     device=h.device), diagonal=1)  # True = masked
        idxs = []
        for blk in self.blocks:
            h, idx = blk(h, mask)
            if idx is not None:
                idxs.append(idx)
        z = self.ln_f(h).mean(dim=1)
        if self.vq_pool is not None:
            z, idx = self.vq_pool(z)
            idxs.append(idx)
        return self.head(z), (idxs if idxs else None)


def train_and_eval(model, train_loader, test_loader, epochs, device):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        train_one_epoch(model, train_loader, opt, device=device)
    return evaluate(model, test_loader, device=device)


# ---------------------------------------------------------------------------
# T1: tightness of the log K upper bound on Transformer representations
# ---------------------------------------------------------------------------

def t1_verdict(results, min_slope=0.7):
    """T1 verdict: produces the conclusion from [(K, logK, H, acc, util), ...] (in ascending K).

    Same criterion as P1/C1: last-segment slope >= 0.7 passes; < 0.5 with low utilization -> codebook collapse.
    Returns (verdict, detail_lines).
    """
    if len(results) < 2:
        return "UNDECIDABLE: fewer than two sweep points", []
    (_, l0, h0, a0, u0), (_, l1, h1, a1, u1) = results[-2], results[-1]
    slope = (h1 - h0) / (l1 - l0)
    detail = [
        f"  last-segment slope = {slope:.2f} (H/logK: {h0 / l0:.2f} -> {h1 / l1:.2f}; "
        f"util: {u0:.0%} -> {u1:.0%})",
        f"  same-direction reading (more robust than slope): last-stage acc {a0:.3f} -> {a1:.3f}",
    ]
    if slope >= min_slope:
        return "PASS: the log K upper bound is likewise tight on Transformer representations (codebook fully used)", detail
    if slope < 0.5 and (u0 < 0.5 or u1 < 0.5):
        return "FAIL: codebook collapse, loose upper bound (check whether reset/metric suit this architecture)", detail
    return "PARTIAL: upper bound loose but no collapse", detail


def verify_t1(device="cpu", epochs=3, quick=False, fast=False):
    """Fixed architecture, sweep the codebook size K of the representation bottleneck, measure whether H(Z) grows logarithmically with log K.

    Criterion: last-segment slope >= 0.7 passes.
    """
    if quick:
        k_list, epochs, subset = (16, 256), 1, 0.1
    elif fast:
        k_list, epochs, subset = (16, 64, 256), 2, 0.25
    else:
        k_list, epochs, subset = (16, 64, 256), 3, None
    print("\n" + "=" * 60)
    print("[T1] log K tightness on Transformer representations: H(Z) grows with log K, slope -> 1")
    print("=" * 60)

    train_loader, test_loader = get_mnist(subset=subset)
    results = []

    for k in k_list:
        torch.manual_seed(k)
        model = TinyGPT(k_pool=k)
        acc, idx_lists = train_and_eval(model, train_loader, test_loader, epochs, device)
        h_z = entropy_of(idx_lists[0], k)
        util = model.vq_pool.codebook_utilization(idx_lists[0])
        log_k = discrete_capacity_logK(k)
        results.append((k, log_k, h_z, acc, util))
        print(f"  K={k:4d}  log K={log_k:5.2f}  H(Z)={h_z:5.2f}  "
              f"H(Z)/logK={h_z / log_k:5.2f}  util={util:4.0%}  acc={acc:.3f}")

    if not quick:
        verdict, detail = t1_verdict(results)
        for line in detail:
            print(line)
        print(f"  [T1 verdict] {verdict}")
    return results


# ---------------------------------------------------------------------------
# T2: the residual stream prevents a narrow bottleneck inside the FFN branch from locking in the accuracy
# ---------------------------------------------------------------------------

def t2_verdict(acc_residual, acc_serial, acc_ceiling, min_gain=0.30):
    """T2 verdict: produces the conclusion from the measured accuracies of the three topologies (independent of training, unit-testable on its own).

    acc_residual: with residuals + a narrow VQ inside each layer's FFN (k_ffn)
    acc_serial  : residuals removed (pure serial) + the same set of narrow VQs
    acc_ceiling : with residuals, no VQ (architecture ceiling reference)

    - success: lift >= min_gain (the residual stream bypasses the bottleneck, the serial min bound fails);
    - failure: lift < 0.10 (removing the residuals makes almost no difference, the min bound still holds on the residual architecture);
    - partial: 0.10 <= lift < min_gain.
    """
    gain = acc_residual - acc_serial
    frac = acc_residual / acc_ceiling if acc_ceiling > 0 else float("nan")
    detail = [
        f"  residual+narrow VQ acc={acc_residual:.3f} | no-residual+same narrow VQs acc={acc_serial:.3f} "
        f"| no-bottleneck ceiling acc={acc_ceiling:.3f}",
        f"  lift = {gain:+.3f}; residual/no-bottleneck-ceiling = {frac:.2f}"
        f" (-> 1 means the bottleneck is genuinely bypassed by the residual stream, no longer an end-to-end constraint)",
    ]
    if gain >= min_gain:
        return ("PASS: the residual stream stops the narrow FFN bottleneck from locking in accuracy -- "
                "the serial min bound fails on a real residual architecture (architecture-level reproduction of P4)"), detail
    if gain < 0.10:
        return "FAIL: removing the residuals makes almost no difference (the min bound still holds on the residual architecture)", detail
    return f"PARTIAL: lift {gain:.3f} falls between 0.10-{min_gain}", detail


def verify_t2(device="cpu", epochs=3, k_ffn=6, n_layer=2, quick=False, fast=False):
    """T2: insert a narrow VQ into each layer's FFN branch and compare with/without residuals.

    Protocol: three arms, same budget, same hyperparameters, **the only independent variable is the residual**:
      (a) with residuals + a narrow VQ with K=k_ffn inside the FFN;
      (b) residuals removed (pure serial) + the same set of narrow VQs;
      (c) with residuals, no VQ (architecture ceiling reference).
    Criterion: lift = acc(a) − acc(b) >= 0.30 passes; < 0.10 fails.

    Honest note: with residuals the bottleneck is bypassed by identity, but the bypass is an **unquantized continuous path**
    whose capacity cannot be given by log K -- so what this experiment proves is "the min bound fails on the residual architecture",
    not "the end-to-end capacity equals Σ log K". This is isomorphic to arm (c) of P4 (P4 has an additional fully discrete arm (b)).
    """
    if quick:
        epochs, subset = 1, 0.1
    elif fast:
        epochs, subset = 2, 0.25
    else:
        epochs, subset = 3, None
    print("\n" + "=" * 60)
    print(f"[T2] residual stream vs pure serial: narrow VQ inserted in the FFN branch (K={k_ffn}x{n_layer} layers)")
    print("=" * 60)

    train_loader, test_loader = get_mnist(subset=subset)

    c_serial = min_capacity_serial([discrete_capacity_logK(k_ffn)] * n_layer)
    print(f"  [capacity] without residuals the {n_layer} narrow VQs are **pure serial**: "
          f"C = min_capacity_serial([log {k_ffn}]x{n_layer}) = {c_serial:.2f} nats -> predicts lock-in")
    print(f"  [topology] with residuals the bottleneck is bypassed by the identity stream, min does not apply -> "
          f"direction should come from sum_capacity_parallel (the bypass is full precision; its capacity is not given by log K)")

    torch.manual_seed(0)
    model_a = TinyGPT(k_ffn=k_ffn, residual=True)
    acc_a, idx_a = train_and_eval(model_a, train_loader, test_loader, epochs, device)
    print(f"  [a residual+narrow VQ] acc={acc_a:.3f}   params {count_params(model_a):6d}   "
          f"H(Z_ffn1)={entropy_of(idx_a[0], k_ffn):.2f} (<= {discrete_capacity_logK(k_ffn):.2f}, always true)")

    torch.manual_seed(0)
    model_b = TinyGPT(k_ffn=k_ffn, residual=False)
    acc_b, idx_b = train_and_eval(model_b, train_loader, test_loader, epochs, device)
    print(f"  [b no-residual+narrow VQ] acc={acc_b:.3f}   params {count_params(model_b):6d}   "
          f"H(Z_ffn1)={entropy_of(idx_b[0], k_ffn):.2f}")

    torch.manual_seed(0)
    model_c = TinyGPT(k_ffn=None, residual=True)
    acc_c, _ = train_and_eval(model_c, train_loader, test_loader, epochs, device)
    print(f"  [c residual, no VQ ] acc={acc_c:.3f}   params {count_params(model_c):6d}")

    if not quick:
        verdict, detail = t2_verdict(acc_a, acc_b, acc_c)
        for line in detail:
            print(line)
        print(f"  [T2 verdict] {verdict}")
    return acc_a, acc_b, acc_c


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Transformer line of Capacity Lens")
    ap.add_argument("--quick", action="store_true", help="quick smoke mode (no verdict output)")
    ap.add_argument("--fast", action="store_true", help="reduced-budget mode, **still outputs a verdict**")
    ap.add_argument("--only", choices=["t1", "t2"], default=None, help="run only the given experiment")
    args = ap.parse_args()
    quick = args.quick
    fast = args.fast and not quick

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode = "quick (smoke, no verdict)" if quick else ("fast (reduced budget, with verdict)" if fast else "full (complete validation)")
    print(f"device: {device}  mode: {mode}")

    if args.only in (None, "t1"):
        verify_t1(device=device, quick=quick, fast=fast)
    if args.only in (None, "t2"):
        verify_t2(device=device, quick=quick, fast=fast)

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
