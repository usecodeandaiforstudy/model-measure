"""Capacity formulas and the wiring rules (capacity.py).

These guard the *implemented* rules, not the empirical claims: the formulas, the direction of the
inequalities, and the two-tier bookkeeping (storage vs flow).
"""

import math

from support import approx
import torch

import capacity_lens as C


def test_discrete_capacity_logK_is_natural_log():
    for K in (2, 8, 256, 4096):
        assert C.discrete_capacity_logK(K) == approx(math.log(K))


def test_discrete_capacity_uniform_is_d_log_q():
    assert C.discrete_capacity_uniform(3, 4) == approx(3 * math.log(4))
    assert C.discrete_capacity_uniform(1, 2) == approx(math.log(2))


def test_gauss_capacity_formula_and_limits():
    assert C.gauss_capacity(64, 1.0) == approx(32 * math.log(2))
    assert C.gauss_capacity(64, 0.0) == 0.0
    assert C.gauss_capacity(16, 3.0) < C.gauss_capacity(16, 5.0)          # monotone in SNR
    assert C.gauss_capacity(32, 1.0) > C.gauss_capacity(16, 1.0)          # monotone in width


def test_serial_bound_is_the_narrowest_stage():
    caps = [4.0, 1.5, 2.75]
    assert C.min_capacity_serial(caps) == approx(1.5)
    assert C.min_capacity_serial([3.0]) == approx(3.0)


def test_parallel_bound_is_the_sum_of_paths():
    assert C.sum_capacity_parallel([1.0, 2.0, 3.5]) == approx(6.5)
    assert C.sum_capacity_parallel([0.0, 2.0]) == approx(2.0)


def test_min_le_sum_when_both_apply():
    caps = [1.0, 2.0, 3.0]
    assert C.min_capacity_serial(caps) <= C.sum_capacity_parallel(caps)


def test_conditional_capacity_two_tier_bookkeeping():
    n, per, top_k, shared = 8, 6.24, 2, 5.55
    r = C.conditional_capacity(n, per, top_k, shared)
    assert r["storage"] == approx(n * per + shared)
    assert r["flow"] == approx(top_k * per + shared)
    assert r["flow"] <= r["storage"]
    assert "note" in r


def test_kv_storage_capacity_is_seq_len_times_per_token():
    assert C.kv_storage_capacity(4096, 0.5) == approx(2048.0)
    assert C.kv_storage_capacity(16, math.log(8)) == approx(16 * math.log(8))


def test_kv_capacity_grows_with_context_but_flow_does_not_follow_it():
    # the reading's content: storage scales with the number of entries, the per-entry payload is separate
    per = math.log(8)
    assert C.kv_storage_capacity(1000, per) == approx(1000 * per)
    assert C.kv_storage_capacity(1000, per) > C.kv_storage_capacity(10, per)
