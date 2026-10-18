"""Criteria for the compression budget advisor (capacity_lens.compress).

Pinned values come from the campaign runs that happened before this module existed, so a change in the
toolkit's arithmetic shows up here rather than in a report. Pure functions only: no dataset, no training,
no checkpoint.

Convention: this module must import and run WITHOUT pytest. The project ships two equivalent runners
(`python tests/run_all.py` and `python -m pytest tests/ -q`), so it uses plain asserts and a small local
`_approx` helper instead of pytest features.
"""
import math

from capacity_lens import (balance_warning, capacity_floors, compress_selftest, eps_for_rib,
                           partition_ceiling, task_info_floor, tightness_interval, veto)


def _approx(got, want, tol):
    assert abs(got - want) <= tol, f"{got!r} != {want!r} (tol {tol})"


def test_selftest_is_clean():
    assert compress_selftest() == []


def test_floor_matches_the_recorded_runs():
    _approx(task_info_floor(0.918, 10, 2.3005), 1.8367, 0.01)     # MNIST, recorded 1.72-1.84
    _approx(task_info_floor(0.753, 10, 2.3024), 1.2009, 0.01)     # CIFAR-10, recorded 1.16-1.20


def test_floor_is_monotone_in_accuracy_and_bounded_by_ln_c():
    assert task_info_floor(0.9, 10) > task_info_floor(0.5, 10)
    _approx(task_info_floor(1.0, 10), math.log(10), 1e-6)   # a perfect model: the floor is ln C
    assert task_info_floor(0.9, 10) <= math.log(10) + 1e-9


def test_interval_brackets_one_and_reproduces_the_recorded_tops():
    mn = tightness_interval(1.97, task_info_floor(0.918, 10, 2.3005), 2.3005)
    cf = tightness_interval(1.60, task_info_floor(0.753, 10, 2.3024), 2.3024)
    _approx(mn[1], 1.252, 0.01)                                   # the flagship certified number
    _approx(cf[1], 1.917, 0.02)
    assert 1.0 <= mn[0] <= mn[1] and 1.0 <= cf[0] <= cf[1]
    assert tightness_interval(None, 1.0, 2.3)[0] == 1.0            # unknown I -> the theorem's own bound


def test_veto_is_sound_and_needs_a_capacity():
    rib = task_info_floor(0.918, 10, 2.3005)
    assert veto(k_codes=4, rib=rib)[0] == "IMPOSSIBLE"             # log 4 < 1.84
    assert veto(k_codes=64, rib=rib)[0] == "FEASIBLE"
    assert veto(capacity_nats=rib - 1e-6, rib=rib)[0] == "IMPOSSIBLE"
    assert veto(capacity_nats=rib, rib=rib)[0] == "FEASIBLE"       # the boundary is inclusive
    assert veto(rib=None)[0] == "UNDECIDABLE"
    assert veto(rib=rib)[0] == "UNDECIDABLE"


def test_balance_warning_fires_only_when_unbalanced():
    assert balance_warning(math.log(10), 10) is None
    assert balance_warning(math.log(10) - 0.005, 10) is None       # inside the tolerance
    assert balance_warning(math.log(10) - 0.5, 10) is not None
    assert "empirical h_y" in balance_warning(2.0, 10)


def test_floors_are_consistent_and_degenerate_on_low_information_tasks():
    fl = capacity_floors(task_info_floor(0.918, 10, 2.3005))
    assert fl["min_codebook_size"] == 7
    assert fl["min_capacity_nats"] > 0
    # MNIST carries only ln 10 nats: the per-dimension bit floor degenerates to one level. That is a
    # recorded property of the bound (feature A), not a defect of this wrapper.
    _approx(fl["quant"][128]["min_levels"], 1.0, 0.05)   # 1.014 levels: degenerate within rounding


def test_partition_ceiling_is_increasing_in_the_retained_information():
    lo, hi = partition_ceiling(0.5, 10), partition_ceiling(2.0, 10)
    assert 0.0 < lo < hi < 1.0
    assert partition_ceiling(0.0, 10) > 0.0


def test_eps_for_rib_inverts_the_floor():
    rib = task_info_floor(0.918, 10, 2.3005)
    eps = eps_for_rib(rib, 10, 2.3005)
    _approx(eps, 0.082, 0.01)
    _approx(task_info_floor(1.0 - eps, 10, 2.3005), rib, 0.01)


# --- the bit-width allocator (zero-training; QAT is a training action and is not shipped here) ------

def test_quantise_weight_is_idempotent_and_untouched_at_high_bits():
    import torch
    from capacity_lens import quantise_weight

    torch.manual_seed(0)
    W = torch.randn(64, 32) * 0.1
    q4 = quantise_weight(W, 4)
    assert torch.allclose(q4, quantise_weight(q4, 4), atol=1e-6)   # already on the grid
    assert quantise_weight(W, 16) is W                             # >= 16 bits is a no-op
    assert not torch.equal(q4, W)                                  # 4 bits really changes the tensor


def test_bit_accounting_is_linear_in_widths():
    import torch
    from capacity_lens import average_bits, total_bits

    model = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.Linear(4, 3))
    total = total_bits(model, [4, 8])
    # only WEIGHTS are quantised (biases are not), so 32*4 + 12*8 bits
    assert total == 32 * 4 + 12 * 8
    _approx(average_bits(model, [8, 8]), 8.0, 1e-12)
    _approx(average_bits(model, [2, 2]), 2.0, 1e-12)


def test_bit_plan_stays_in_budget_and_restores_the_model():
    import torch
    from capacity_lens import BIT_GRID, accuracy_at_bits, plan_bit_allocation

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
    x, y = torch.randn(256, 16), torch.randint(0, 4, (256,))
    res = plan_bit_allocation(model, x, y, 3.0)
    assert set(res["arms"]) == {"uniform", "greedy", "mixed"}
    for bits in res["arms"].values():
        assert len(bits) == 2 and all(b in BIT_GRID for b in bits)
    for name in res["arms"]:
        assert res["average"][name] <= 3.0 + 1e-9      # never over the budget
        assert 0.0 <= res["accuracy"][name] <= 1.0     # a measurable accuracy
    with torch.no_grad():
        before = [m.weight.data.clone() for m in model.modules() if hasattr(m, "weight")]
    accuracy_at_bits(model, x, y, [2, 2])
    with torch.no_grad():
        after = [m.weight.data for m in model.modules() if hasattr(m, "weight")]
    assert all(torch.equal(b, a) for b, a in zip(before, after))   # the model is restored
