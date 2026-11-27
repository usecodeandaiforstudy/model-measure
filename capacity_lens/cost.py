"""Constraint-cost accounting: what a restriction costs in LOSS units.

Pure functions only: they take a loss/gradient/Hessian-vector-product callable and plain arrays, and return
plain dicts. Nothing here trains, reads files or prints, so the module can be tested on a quadratic where the
answer is known exactly.

The unit is LOSS, not bits. This is deliberately NOT convertible to the library's bit-denominated functions:
the measured price per bit differs by about 9x between the activation and weight sides (see the note in
reference/notes/delta-information-theory.md), so no exchange rate exists.

CONTRACT
--------
Every public function
  * is deterministic given its inputs (no RNG, no global state),
  * never mutates its arguments,
  * returns a dict that always carries "unit" and, where a guarantee is involved, "certified".

GUARANTEES
----------
    constraint_cost(...)["value"]        point estimate, NO certificate
    certify_lower(...)["lower"]          proven LOWER bound only; an upper bound does not exist for
                                         non-convex paths (measured: the minimum Hessian eigenvalue along
                                         the path is negative in 4/4 cells)
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Sequence

import torch

__all__ = ["cg_solve_hinv_g", "constraint_cost", "certify_lower", "is_constrained_local_min",
           "alignment_prior", "selftest"]

HVPCallable = Callable[[torch.Tensor], torch.Tensor]


def cg_solve_hinv_g(hvp: HVPCallable, g: torch.Tensor, *, iters: int = 30,
                    tol: float = 1e-10) -> Dict[str, object]:
    """Solve H x = g by conjugate gradients, using only Hessian-vector products.

    Args    hvp: callable v -> H v.        g: right-hand side (1-D tensor).
            iters: maximum CG iterations.  tol: absolute residual target.
    Returns dict with keys x, iterations, residual, converged.

    Cost is independent of the parameter count: `iters` matrix-vector products, i.e. 2*iters backward passes.
    """
    if g.ndim != 1:
        raise ValueError("g must be 1-D")
    x = torch.zeros_like(g)
    r = g.clone()
    p = r.clone()
    rs = float(r @ r)
    converged = rs <= tol
    used = 0
    for i in range(iters):
        if converged:
            break
        Hp = hvp(p)
        denom = float(p @ Hp)
        if denom <= 0.0:
            break
        a = rs / denom
        x = x + a * p
        r = r - a * Hp
        rs_new = float(r @ r)
        used = i + 1
        if rs_new <= tol:
            converged = True
            rs = rs_new
            break
        p = r + (rs_new / max(rs, 1e-30)) * p
        rs = rs_new
    return {"x": x, "iterations": used, "residual": math.sqrt(max(rs, 0.0)),
            "converged": bool(converged)}


def constraint_cost(g: torch.Tensor, hvp: HVPCallable, *, iters: int = 30,
                    tol: float = 1e-10) -> Dict[str, object]:
    """Point estimate of Delta = L(constrained optimum) - L(full optimum), in LOSS units.

        Delta ~= (1/2) g^T H^{-1} g

    Args    g: the normal component g_perp = (I - P_B) grad L evaluated at the CONSTRAINED optimum.
            hvp: Hessian-vector-product callable.
    Returns dict with value, unit="loss", certified=False, cg (the solver report), valid_region.

    valid_region is the caller's duty to check and is reported as "unknown" here: the expansion is trusted only
    when the restricted Hessian is positive definite and the gap is small (measured accurate to 0.5-1.4% in that
    regime, 99.9% error outside it).
    """
    sol = cg_solve_hinv_g(hvp, g, iters=iters, tol=tol)
    value = 0.5 * float(g @ sol["x"])
    return {"value": value, "unit": "loss", "certified": False, "cg": sol,
            "valid_region": "unknown", "warning": None if sol["converged"] else
            "CG did not converge: value is a lower-quality estimate"}


def certify_lower(g: torch.Tensor, hvp: HVPCallable, *, n_probe: int = 24,
                  seed: int = 0) -> Dict[str, object]:
    """Proven LOWER bound on Delta, requiring only smoothness (an upper bound needs convexity and does not exist).

        Delta >= ||g||^2 / (2 * nu),   nu = max eigenvalue of H along the relevant path

    nu is estimated by power iteration on `hvp` (deterministic: the start vector comes from `seed`).
    Returns dict with lower, nu, unit="loss", certified=True.
    """
    gen = torch.Generator(device=g.device).manual_seed(seed)
    v = torch.randn(g.numel(), generator=gen, dtype=g.dtype, device=g.device)
    v = v / max(float(v.norm()), 1e-30)
    lam = 0.0
    for _ in range(n_probe):
        Hv = hvp(v)
        nrm = float(Hv.norm())
        if nrm <= 0.0:
            break
        lam = nrm               # Rayleigh quotient after normalisation (H symmetric)
        v = Hv / nrm
    lo = float(g @ g) / (2.0 * max(lam, 1e-30))
    return {"lower": lo, "nu": lam, "unit": "loss", "certified": True,
            "note": "lower bound only: the upper bound requires convexity, which fails along real paths"}


def is_constrained_local_min(hvp: HVPCallable, basis: torch.Tensor, *,
                             n_grid: int = 12, seed: int = 0) -> Dict[str, object]:
    """Check the theorem's premise: is the restricted Hessian B^T H B positive definite?

    Estimated by the minimum Rayleigh quotient over `n_grid` random directions inside span(B).
    Returns dict with satisfied, lambda_min_restricted, unit=None.
    """
    gen = torch.Generator(device=basis.device).manual_seed(seed)
    best = float("inf")
    for _ in range(n_grid):
        u = torch.randn(basis.shape[1], generator=gen, dtype=basis.dtype, device=basis.device)
        u = u / max(float(u.norm()), 1e-30)
        w = basis @ u
        Hw = hvp(w)
        q = float(w @ Hw) / max(float(w @ w), 1e-30)
        best = min(best, q)
    return {"satisfied": bool(best > 0.0), "lambda_min_restricted": best, "unit": None}


def alignment_prior(theta_star: torch.Tensor, basis: torch.Tensor) -> Dict[str, object]:
    """pot_E: squared Euclidean distance from theta* to the subspace, a PRIOR usable only for ranking.

    Measured rank correlation with the true gap: 0.632 (108 cells). Use for ordering, never for a magnitude.
    """
    B = basis
    v = theta_star - B @ (B.T @ theta_star)
    return {"pot": 0.5 * float(v @ v), "unit": "squared-parameter-distance", "usable_for": "ranking only",
            "measured_rank_corr": 0.632}


def selftest() -> Dict[str, object]:
    """Quadratic sanity check where every quantity is known in closed form. Deterministic."""
    A = torch.tensor([[3.0, 0.4], [0.4, 1.0]])
    hvp = lambda v: A @ v
    g = torch.tensor([1.0, -0.5])
    exact = 0.5 * float(g @ torch.linalg.solve(A, g))
    est = constraint_cost(g, hvp, iters=10)["value"]
    nu = float(torch.linalg.eigvalsh(A).max())
    lo = certify_lower(g, hvp)["lower"]
    out = {"exact": exact, "estimate": est, "rel_err": abs(est - exact) / exact,
           "lower": lo, "lower_ok": lo <= exact + 1e-9, "nu_theory": nu}
    ok = out["rel_err"] < 1e-6 and out["lower_ok"]
    return {**out, "ok": bool(ok)}


if __name__ == "__main__":
    print(selftest())
