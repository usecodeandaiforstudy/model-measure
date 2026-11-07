"""Codec measurements: what an encoder delivers, what a decoder leaves on the table, and where the loss is.

Only reads capacity_lens (no formula is reimplemented).

ATTRIBUTION RULE (validated on controlled cases in reference/scripts/r16_codec_attribution.py)
----------------------------------------------------------------------------------------------
The reference is EMPIRICAL: a strong model trained on the raw input (`reference_acc`). An earlier version
used the accuracy ceiling derived from the probe's I_lb, but because I_lb is a LOWER bound that "ceiling"
was violated by the measurement itself (predicted 0.500, achieved 0.643), so it is never used as a bound.

    acc_strong <= reference_acc - 0.10  -> "LOSS IN ENCODER"
    acc_strong - acc_actual > 0.10      -> "LOSS IN DECODER"
    otherwise                           -> "NEITHER"

Without a reference the report returns "UNDECIDED" instead of guessing.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

import capacity_lens as CL

__all__ = ["latent_report", "codec_report"]


def _mlp(d_in, c, hidden=256):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, c))


def _train(model, Z, y, epochs=20, lr=3e-3, bs=64, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        perm = torch.randperm(len(Z))
        for i in range(0, len(Z), bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            lossf(model(Z[j]), y[j]).backward()
            opt.step()
    return model


def _acc(model, Z, y):
    with torch.no_grad():
        return float((model(Z).argmax(1) == y).float().mean())


def _hz(Z, bins=8):
    """Binned entropy of the latent. Reliable only for latent dim <= ~6 (beyond that it saturates at ln n)."""
    with torch.no_grad():
        lo, hi = Z.min(0).values, Z.max(0).values
        span = (hi - lo).clamp_min(1e-9)
        idx = ((Z - lo) / span * bins).clamp(0, bins - 1).to(torch.int64)
        mult = bins ** torch.arange(idx.shape[1], dtype=torch.int64)
        codes = (idx * mult).sum(1)
        _, counts = torch.unique(codes, return_counts=True)
        p = counts.float() / counts.sum()
        return float(-(p * p.log()).sum())


def latent_report(Z, y, seed=0):
    """Measurements that need only the latent and the labels: capacity, information, efficiency."""
    Z = Z if isinstance(Z, torch.Tensor) else torch.as_tensor(Z, dtype=torch.float32)
    y = y if isinstance(y, torch.Tensor) else torch.as_tensor(y)
    Z, y = Z.detach().float(), y.long()
    c = int(y.max()) + 1
    h = len(Z) // 2
    i_lb = float(CL.probe_lower_bound(Z[:h], y[:h], Z[h:], y[h:], c, seed=seed))
    hz = _hz(Z)
    eff = (i_lb / hz) if hz > 0 else float("nan")
    out = dict(latent=int(Z.shape[1]), c=c, h_z=hz, i_lb=i_lb, efficiency=eff,
               h_z_reliable=bool(Z.shape[1] <= 6),
               unavailable=[] if len(Z) >= 4 * c * 8 else
               ["I_lb: fewer than 4*K*C samples per half-split; treat the probe as noisy"])
    if not out["h_z_reliable"]:
        out["unavailable"].append("H(Z): latent dimension above ~6 makes the binned entropy saturate")
    return out


def codec_report(enc, dec, X, y, reference_acc=None, seed=0, epochs=20):
    """Attribution of the loss between encoder and decoder. `reference_acc` is a float or a callable.

    Without `reference_acc` the verdict is "UNDECIDED" -- the rule needs an empirical reference and this
    function will not substitute an information-theoretic bound for one.
    """
    X = X if isinstance(X, torch.Tensor) else torch.as_tensor(X, dtype=torch.float32)
    y = y if isinstance(y, torch.Tensor) else torch.as_tensor(y)
    X, y = X.detach().float(), y.long()
    with torch.no_grad():
        z = enc(X)
        z = z[0] if isinstance(z, tuple) else z
    z = z.detach()
    c = int(y.max()) + 1
    lat = latent_report(z, y, seed=seed)
    a_actual = _acc(dec, z, y)
    strong = _train(_mlp(z.shape[1], c), z, y, epochs=epochs, seed=seed + 1)
    a_strong = _acc(strong, z, y)
    gap = a_strong - a_actual
    if reference_acc is None:
        verdict, ceiling = "UNDECIDED (no empirical reference supplied)", None
    else:
        ceiling = float(reference_acc() if callable(reference_acc) else reference_acc)
        if a_strong <= ceiling - 0.10:
            verdict = "LOSS IN ENCODER"
        elif gap > 0.10:
            verdict = "LOSS IN DECODER"
        else:
            verdict = "NEITHER"
    return dict(acc_actual=a_actual, acc_strong=a_strong, gap=gap, reference_acc=ceiling,
                verdict=verdict, **lat,
                notes=["reference must be measured on the raw input; I_lb-derived ceilings are invalid",
                       "H(Z) is a capacity upper bound and is only reliable for small latents"])
