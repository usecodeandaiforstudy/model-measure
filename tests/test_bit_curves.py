"""Tests for capacity_lens_ext.bit_curves. Exact arithmetic on supplied curves."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from capacity_lens.bit_curves import DEFAULT_CLIFF_RATIO, activation_curve, pruning_curve, selftest


def test_activation_delta_is_relative_to_baseline():
    r = activation_curve({2: 1.5}, 1.0)
    assert abs(r["delta"][2] - 0.5) < 1e-12 and r["units"] == "loss"


def test_activation_saturation_point():
    r = activation_curve({1: 0.08, 2: 0.02, 3: 0.005, 4: 0.001}, 0.0)
    assert r["saturated_at"] == 3          # tol 1e-2 => |delta|<=0.01 from b=3 on


def test_activation_never_saturates():
    r = activation_curve({1: 1.0, 2: 0.9, 4: 0.8}, 0.0)
    assert r["saturated_at"] is None


def test_activation_monotone_detection():
    assert activation_curve({1: 1.0, 2: 0.5}, 0.0)["monotone"] is True
    assert activation_curve({1: 0.5, 2: 0.9}, 0.0)["monotone"] is False


def test_activation_rejects_negative_bits():
    try:
        activation_curve({-1: 1.0}, 0.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_activation_rejects_empty():
    try:
        activation_curve({}, 0.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_pruning_cliff_detected():
    r = pruning_curve({1: 0.003, 2: 0.005, 3: 0.021, 4: 0.264}, 0.0)
    assert r["cliff_k"] == 4 and r["safe_k"] == 3


def test_pruning_no_cliff_when_smooth():
    r = pruning_curve({1: 0.01, 2: 0.02, 3: 0.03, 4: 0.04}, 0.0)
    assert r["cliff_k"] is None and r["safe_k"] is None


def test_pruning_monotone_detection():
    assert pruning_curve({1: 0.01, 2: 0.02}, 0.0)["monotone"] is True
    assert pruning_curve({1: 0.05, 2: 0.01}, 0.0)["monotone"] is False


def test_pruning_cliff_ratio_threshold():
    # ratio 1.9 < 5 => no cliff; ratio 10 >= 5 => cliff
    lo = pruning_curve({1: 0.01, 2: 0.019}, 0.0)
    hi = pruning_curve({1: 0.01, 2: 0.10}, 0.0)
    assert lo["cliff_k"] is None and hi["cliff_k"] == 2
    assert DEFAULT_CLIFF_RATIO == 5.0


def test_pruning_rejects_empty():
    try:
        pruning_curve({}, 0.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


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
