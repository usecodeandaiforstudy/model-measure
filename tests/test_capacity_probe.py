"""Tests for capacity_lens_ext.capacity_probe. Fake fit callbacks => every assertion is an exact number."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from capacity_lens.capacity_probe import (DRAWS_MIN, _fake_fit, bisect_threshold, measure_capacity, p_fit, selftest)


def test_pfit_all_success():
    assert p_fit(_fake_fit(1000), 10, steps=5, draws=5) == 1.0


def test_pfit_none():
    assert p_fit(_fake_fit(10), 100, steps=5, draws=5) == 0.0


def test_pfit_partial_is_exact():
    calls = {"n": 0}

    def alternating(n, steps, seed):
        calls["n"] += 1
        return calls["n"] % 2 == 0          # 2 of 4 would be 0.5, but draws must be >= 5

    assert abs(p_fit(alternating, 5, steps=1, draws=6) - 0.5) < 1e-12


def test_pfit_rejects_few_draws():
    try:
        p_fit(_fake_fit(10), 5, steps=1, draws=DRAWS_MIN - 1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for draws < 5 (rule 3)")


def test_pfit_rejects_zero_n():
    try:
        p_fit(_fake_fit(10), 0, steps=1, draws=5)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_bisect_exact_at_grid_point():
    r = bisect_threshold(_fake_fit(300), steps=1, draws=5, grid=(100, 200, 300, 400, 600))
    assert abs(float(r["threshold"]) - 300.0) < 1e-9 and r["crossed"] is True


def test_bisect_returns_largest_passing_point():
    # a model whose limit is 150 fails at 200, so the largest grid point that still passes is 100
    r = bisect_threshold(_fake_fit(150), steps=1, draws=5, grid=(100, 200, 400))
    assert float(r["threshold"]) == 100.0 and r["crossed"] is True


def test_bisect_boundary_is_not_crossed():
    r = bisect_threshold(_fake_fit(10 ** 6), steps=1, draws=5, grid=(100, 200))
    assert r["crossed"] is False


def test_measure_capacity_reports_capacity_and_rules():
    rep = measure_capacity(_fake_fit(300), budget=7, draws=5, grid=(100, 200, 300, 400, 600),
                           arch={"d_in": 10, "h": 16, "act": "relu"})
    assert abs(float(rep["capacity"]) - 300.0) < 1e-9
    assert rep["rules"]["draws"] == 5 and rep["rules"]["architecture_reported"] is True
    assert rep["rules"]["budget"] == 14


def test_measure_capacity_saturated_true():
    rep = measure_capacity(_fake_fit(300), budget=10, draws=5, grid=(100, 200, 300, 400, 600))
    assert rep["saturated"] is True and rep["lower_bound_only"] is False and rep["warning"] is None


def test_measure_capacity_unsaturate_flagged():
    def budget_dependent(n, steps, seed):
        return n <= 100 + 10 * steps          # threshold grows with the budget => never saturates

    rep = measure_capacity(budget_dependent, budget=10, draws=5, grid=(100, 200, 300, 400, 600))
    assert rep["saturated"] is False
    assert rep["lower_bound_only"] is True
    assert isinstance(rep["warning"], str) and "NOT saturated" in rep["warning"]


def test_measure_capacity_rejects_few_draws():
    try:
        measure_capacity(_fake_fit(100), budget=5, draws=3)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_grid_pfit_echoed():
    rep = measure_capacity(_fake_fit(300), budget=5, draws=5, grid=(100, 300, 500))
    assert len(rep["grid_pfit"]["B"]) == 3 and len(rep["grid_pfit"]["2B"]) == 3


def test_selftest_passes():
    assert selftest()["ok"] is True


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
