"""Tests for capacity_lens_ext.cost. Fixed numbers only: no tolerance-free assertions on random data."""
from __future__ import annotations

import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from capacity_lens.cost import (alignment_prior, cg_solve_hinv_g, certify_lower, constraint_cost,
                                    is_constrained_local_min, selftest)


def _diag(vals):
    A = torch.diag(torch.tensor(vals, dtype=torch.float64))
    return A, (lambda v: A @ v)


def test_cg_is_exact_on_diagonal():
    A, hvp = _diag([2.0, 4.0, 8.0])
    g = torch.tensor([2.0, 4.0, 8.0], dtype=torch.float64)
    sol = cg_solve_hinv_g(hvp, g, iters=3)
    assert torch.allclose(sol["x"], torch.ones(3, dtype=torch.float64), atol=1e-10)
    assert sol["converged"] is True


def test_cg_identity_needs_one_iteration():
    I = torch.eye(4, dtype=torch.float64)
    g = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    sol = cg_solve_hinv_g(lambda v: I @ v, g, iters=5)
    assert sol["iterations"] == 1


def test_constraint_cost_quadratic_exact():
    A, hvp = _diag([1.0, 2.0, 4.0])
    g = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    exact = 0.5 * float(sum(gi * gi / ai for gi, ai in zip(g.tolist(), [1.0, 2.0, 4.0])))
    got = constraint_cost(g, hvp, iters=20)["value"]
    assert abs(got - exact) / exact < 1e-9


def test_unit_and_certification_flags():
    A, hvp = _diag([1.0, 1.0])
    g = torch.ones(2, dtype=torch.float64)
    r = constraint_cost(g, hvp)
    assert r["unit"] == "loss" and r["certified"] is False


def test_lower_bound_holds():
    A, hvp = _diag([1.0, 9.0])
    g = torch.tensor([1.0, 1.0], dtype=torch.float64)
    exact = 0.5 * float(g[0] ** 2 / 1.0 + g[1] ** 2 / 9.0)
    lo = certify_lower(g, hvp)["lower"]
    assert lo <= exact + 1e-9


def test_lower_bound_is_positive():
    A, hvp = _diag([1.0, 2.0])
    g = torch.tensor([1.0, 1.0], dtype=torch.float64)
    assert certify_lower(g, hvp)["lower"] > 0.0


def test_lower_bound_is_loose_on_ill_conditioned():
    A, hvp = _diag([1.0, 100.0])
    g = torch.tensor([1.0, 1.0], dtype=torch.float64)
    exact = 0.5 * (1.0 + 1.0 / 100.0)
    assert certify_lower(g, hvp)["lower"] < exact


def test_restricted_min_true_for_pd():
    A, hvp = _diag([1.0, 2.0, 3.0])
    B = torch.eye(3, dtype=torch.float64)
    r = is_constrained_local_min(hvp, B)
    assert r["satisfied"] is True and r["lambda_min_restricted"] > 1.0 - 1e-9


def test_restricted_min_false_for_indefinite():
    A, hvp = _diag([-1.0, 2.0])
    B = torch.eye(2, dtype=torch.float64)
    assert is_constrained_local_min(hvp, B)["satisfied"] is False


def test_alignment_prior_zero_when_theta_in_subspace():
    B = torch.eye(3, dtype=torch.float64)
    th = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)
    assert alignment_prior(th, B)["pot"] < 1e-20


def test_alignment_prior_positive_outside_subspace():
    B = torch.zeros(3, 1, dtype=torch.float64); B[0, 0] = 1.0
    th = torch.tensor([0.0, 3.0, 4.0], dtype=torch.float64)
    assert abs(alignment_prior(th, B)["pot"] - 12.5) < 1e-12


def test_selftest_passes():
    assert selftest()["ok"] is True


def test_cg_rejects_2d():
    A, hvp = _diag([1.0, 1.0])
    try:
        cg_solve_hinv_g(hvp, torch.ones(2, 2, dtype=torch.float64))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for f in fns:
        try:
            f(); print(f"  PASS {f.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1; print(f"  FAIL {f.__name__}: {e}")
    print(f"  {len(fns) - bad}/{len(fns)} passed")
    raise SystemExit(1 if bad else 0)
