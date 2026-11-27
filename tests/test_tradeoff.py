"""Tests for capacity_lens_ext.tradeoff. Exact arithmetic on supplied curves."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from capacity_lens.tradeoff import codec_tradeoff, selftest

PCA = {50: 0.01047, 100: 0.00476, 200: 0.00176, 400: 0.00022}
AE = {50: 0.00483, 100: 0.00333, 200: 0.00378, 400: 0.00296}


def test_crossover_on_measured_curves():
    assert codec_tradeoff(PCA, AE)["crossover_k"] == 200


def test_winner_below_and_above_crossover():
    w = codec_tradeoff(PCA, AE)["winner_at"]
    assert w[50] == "ae" and w[100] == "ae" and w[200] == "pca" and w[400] == "pca"


def test_ae_floor_is_minimum():
    assert abs(codec_tradeoff(PCA, AE)["ae_floor"] - 0.00296) < 1e-12


def test_delta_sign_convention():
    d = codec_tradeoff(PCA, AE)["delta_pca_minus_ae"]
    assert d[50] > 0 and d[200] < 0


def test_shared_k_intersection_only():
    r = codec_tradeoff({10: 1.0, 20: 0.5}, {20: 0.4, 30: 0.3})
    assert r["shared_k"] == [20] and r["winner_at"] == {20: "ae"}


def test_no_shared_rate_raises():
    try:
        codec_tradeoff({10: 1.0}, {20: 1.0})
    except ValueError:
        return
    raise AssertionError("expected ValueError when the curves share no rate point")


def test_ae_never_wins_gives_crossover_at_first():
    r = codec_tradeoff({10: 0.1, 20: 0.01}, {10: 0.5, 20: 0.4})
    assert r["crossover_k"] == 10 and r["ae_max_advantage_pct"] is None


def test_ae_always_wins_gives_none():
    r = codec_tradeoff({10: 0.5, 20: 0.4}, {10: 0.1, 20: 0.01})
    assert r["crossover_k"] is None and r["ae_max_advantage_pct"] is not None


def test_max_advantage_pct_is_exact():
    r = codec_tradeoff({10: 1.0}, {10: 0.5})
    assert abs(r["ae_max_advantage_pct"] - 50.0) < 1e-12


def test_recommendation_present():
    r = codec_tradeoff(PCA, AE)
    assert isinstance(r["recommendation"], str) and "floor" in r["recommendation"]


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
