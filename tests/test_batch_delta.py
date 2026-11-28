"""Tests for capacity_lens_ext.batch_delta. Exact numbers only."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from capacity_lens.batch_delta import (MEASURED_CV, ONLINE_BATCH_CV, ONLINE_TRACK_SPEARMAN, PROBE_SPEARMAN,
                                           delta_summary, online_step_delta, per_batch_delta, probe_signal,
                                           rank_batches, select_batches, selftest, track_probe)

FULL = 0.5
BO = {0: 0.7, 1: 0.6, 2: 0.4, 3: 1.0}


def test_per_batch_delta_subtracts_reference():
    d = per_batch_delta(BO, FULL)
    assert abs(d[0] - 0.2) < 1e-12 and abs(d[2] + 0.1) < 1e-12


def test_per_batch_delta_rejects_nan_reference():
    try:
        per_batch_delta(BO, float("nan"))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_summary_mean_is_exact():
    d = per_batch_delta(BO, FULL)
    s = delta_summary(d)
    assert abs(s["mean"] - 0.175) < 1e-12   # (0.2 + 0.1 - 0.1 + 0.5) / 4


def test_summary_identifies_extremes():
    s = delta_summary(per_batch_delta(BO, FULL))
    assert s["max_batch"] == 3 and s["min_batch"] == 2


def test_summary_rejects_empty():
    try:
        delta_summary({})
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_rank_is_deterministic_and_sorted():
    d = per_batch_delta(BO, FULL)
    assert rank_batches(d)[0][0] == 3 and rank_batches(d)[-1][0] == 2
    assert rank_batches(d) == rank_batches(d)


def test_select_high_and_low():
    d = per_batch_delta(BO, FULL)
    assert select_batches(d, frac=0.5, mode="high") == [0, 3]   # ids ascending, as documented
    assert select_batches(d, frac=0.5, mode="low") == [1, 2]


def test_select_random_is_reproducible():
    d = per_batch_delta(BO, FULL)
    assert select_batches(d, frac=0.5, mode="random", seed=7) == \
        select_batches(d, frac=0.5, mode="random", seed=7)


def test_select_middle_has_right_size():
    d = per_batch_delta(BO, FULL)
    assert len(select_batches(d, frac=0.5, mode="middle")) == 2


def test_select_rejects_bad_frac_and_mode():
    d = per_batch_delta(BO, FULL)
    for bad in (0.0, 1.5):
        try:
            select_batches(d, frac=bad, mode="high")
        except ValueError:
            continue
        raise AssertionError("expected ValueError for frac")
    try:
        select_batches(d, frac=0.5, mode="nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError for mode")


def test_probe_signal_direction_and_value():
    r = probe_signal([1.0, 3.0, 9.0], window=2)
    assert abs(r["early_grad_norm"] - 2.0) < 1e-12 and r["direction"] == "higher is worse"
    assert r["measured_spearman"] == PROBE_SPEARMAN


def test_probe_signal_rejects_empty():
    try:
        probe_signal([])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_measured_constants_are_recorded():
    assert abs(MEASURED_CV - 0.213) < 1e-12


def test_selftest_passes():
    assert selftest()["ok"] is True


def test_online_step_delta_sign():
    # <g_probe, g_b> > 0 means the batch update LOWERS the probe loss, so delta_step is negative
    r = online_step_delta(grad_batch_probe_dot=2.0, lr=0.5)
    assert abs(r["delta_step"] + 1.0) < 1e-12 and r["detector_only"] is True


def test_online_step_delta_rejects_bad_lr():
    try:
        online_step_delta(grad_batch_probe_dot=1.0, lr=0.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_track_probe_spread_ratio():
    r = track_probe([-1.0, -1.0, -1.0], [0.5])
    assert abs(r["spread_ratio"]) < 1e-12


def test_track_probe_epoch_tracking_is_negative():
    # as the model converges the per-step probe improvement SHRINKS (deltas go -5 -> -1) while the probe loss
    # falls (1.0 -> 0.6), so the correlation between the two series is -1 (measured -0.969 in R116)
    r = track_probe([-5.0, -5.0, -3.0, -3.0, -1.0, -1.0], [1.0, 0.8, 0.6])
    assert r["epoch_tracking_spearman"] is not None and r["epoch_tracking_spearman"] < -0.9


def test_track_probe_rejects_empty():
    try:
        track_probe([], [1.0])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_online_constants_recorded():
    assert abs(ONLINE_BATCH_CV - 0.10) < 1e-12 and abs(ONLINE_TRACK_SPEARMAN + 0.969) < 1e-12


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
