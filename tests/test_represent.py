"""Tests for capacity_lens_ext.represent. Exact arithmetic, no randomness."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from capacity_lens.represent import (MIN_BASELINES, WINDOW_STEPS, align_report, calibrate_threshold,
                                         selftest, subset_rho_prime)


def test_tau_floor_applies():
    assert abs(calibrate_threshold([1.0] * 12)["tau"] - 2.0) < 1e-12


def test_tau_uses_q95_when_larger():
    # 19 ones and one 10.0 => interpolated q95 = 1.45, so 1.5*q95 = 2.175 beats the 2.0 floor
    base = [1.0] * 19 + [10.0]
    cal = calibrate_threshold(base)
    assert abs(cal["q95"] - 1.45) < 1e-12
    assert abs(cal["tau"] - 2.175) < 1e-12 and cal["tau"] > cal["floor"]


def test_tau_stable_flag():
    assert calibrate_threshold([1.0] * MIN_BASELINES)["stable"] is True
    assert calibrate_threshold([1.0] * (MIN_BASELINES - 1))["stable"] is False


def test_tau_note_present_when_unstable():
    cal = calibrate_threshold([1.0] * 3)
    assert isinstance(cal["note"], str) and "not stable" in cal["note"]


def test_calibrate_rejects_empty():
    try:
        calibrate_threshold([])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_calibrate_rejects_nonpositive():
    try:
        calibrate_threshold([1.0, 0.0])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_rho_prime_normaliser_is_median():
    r = subset_rho_prime({"a": 2.0}, [1.0, 1.0, 3.0])
    assert abs(r["normaliser"] - 1.0) < 1e-12 and abs(r["rho_prime"]["a"] - 2.0) < 1e-12


def test_rho_prime_is_dimensionless():
    r1 = subset_rho_prime({"a": 2.0}, [1.0] * 10)
    r2 = subset_rho_prime({"a": 20.0}, [10.0] * 10)
    assert abs(r1["rho_prime"]["a"] - r2["rho_prime"]["a"]) < 1e-12


def test_verdict_split_at_tau():
    rep = align_report({"x": 1.99, "y": 2.01}, [1.0] * 12)
    got = {s["name"]: s["verdict"] for s in rep["subsets"]}
    assert got["x"] == "representative" and got["y"] == "not-representative"


def test_report_always_has_window_steps():
    rep = align_report({"a": 1.0}, [1.0] * 12)
    assert rep["validity"]["window_steps"] == WINDOW_STEPS == 100


def test_report_boundaries_nonempty():
    rep = align_report({"a": 1.0}, [1.0] * 12)
    assert len(rep["boundaries"]) >= 4


def test_report_subsets_sorted_by_name():
    rep = align_report({"b": 1.0, "a": 2.0, "c": 3.0}, [1.0] * 12)
    assert [s["name"] for s in rep["subsets"]] == ["a", "b", "c"]


def test_report_custom_window():
    rep = align_report({"a": 1.0}, [1.0] * 12, window_steps=50)
    assert rep["validity"]["window_steps"] == 50


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
