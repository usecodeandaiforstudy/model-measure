"""Encoder choice: at a given rate, is a nonlinear autoencoder better than PCA?

Pure functions only: both rate-distortion curves are supplied by the caller as k -> reconstruction MSE, so the
module is arithmetic. Every test is an exact number, and the selftest uses the curves actually measured on MNIST.

MEASURED SHAPE (reference/notes/encoder-decoder-delta.md)
    k=50   PCA 0.01047  AE 0.00483   -> AE wins by 54%
    k=100  PCA 0.00476  AE 0.00333   -> AE wins by 30%
    k=200  PCA 0.00176  AE 0.00378   -> PCA wins by 115%
    k=400  PCA 0.00022  AE 0.00296   -> PCA wins by 1260%
    the AE has a FLOOR (~0.0030) while PCA tends to 0, so the crossover is real and cannot be removed by a wider
    bottleneck (a 512-wide AE moved it from ~50-100 to ~100-200 and no further).
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

__all__ = ["codec_tradeoff", "selftest"]


def codec_tradeoff(pca: Mapping[int, float], ae: Mapping[int, float]) -> Dict[str, object]:
    """Locate the crossover and the AE floor from two rate-distortion curves.

    `crossover_k` is the smallest shared k at which the AE is no longer better than PCA (>= comparison, so ties
    count as a crossover). `winner_at` gives the winner for every shared k. `recommendation` states the
    operational rule: use the AE below the crossover, PCA above it.

    Raises ValueError when the curves share no k, because a crossover is meaningless without shared rates.
    """
    shared = sorted(set(int(k) for k in pca) & set(int(k) for k in ae))
    if not shared:
        raise ValueError("the two curves share no rate point")
    delta = {k: float(pca[k]) - float(ae[k]) for k in shared}          # >0 means the AE is better
    winner = {k: ("ae" if delta[k] > 0 else "pca") for k in shared}
    crossover: Optional[int] = None
    for k in shared:
        if delta[k] <= 0.0:
            crossover = k
            break
    ae_floor = min(float(v) for v in ae.values())
    pca_best = min(float(v) for v in pca.values())
    low = [k for k in shared if delta[k] > 0]
    low_win_pct = (100.0 * max(delta[k] / float(pca[k]) for k in low)) if low else None
    return {
        "shared_k": shared,
        "delta_pca_minus_ae": delta,
        "winner_at": winner,
        "crossover_k": crossover,
        "ae_floor": ae_floor,
        "pca_best": pca_best,
        "ae_max_advantage_pct": low_win_pct,
        "units": "loss (reconstruction MSE)",
        "recommendation": ("use the autoencoder below the crossover and PCA above it: the AE has a distortion "
                           "floor, PCA tends to zero"),
    }


def selftest() -> Dict[str, object]:
    """Exact check against the measured MNIST curves: crossover must be 200 and the AE floor 0.00296."""
    pca = {50: 0.01047, 100: 0.00476, 200: 0.00176, 400: 0.00022}
    ae = {50: 0.00483, 100: 0.00333, 200: 0.00378, 400: 0.00296}
    r = codec_tradeoff(pca, ae)
    ok = (r["crossover_k"] == 200 and abs(r["ae_floor"] - 0.00296) < 1e-12
          and r["winner_at"][50] == "ae" and r["winner_at"][400] == "pca")
    return {"crossover_k": r["crossover_k"], "ae_floor": r["ae_floor"], "ok": bool(ok)}


if __name__ == "__main__":
    print(selftest())
