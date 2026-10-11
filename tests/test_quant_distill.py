"""The quantization and distillation screens (quant.py, distill.py), plus the Fano inversion."""

import math

from support import approx

import capacity_lens as C


def test_quant_min_levels_is_two_pow_rib_over_d():
    q = C.QuantCapacity()
    assert q.min_levels(6.0, 3) == approx(2.0 ** 2.0)              # the documented base is 2, not e
    assert q.min_levels(3.3, 32) == approx(2.0 ** (3.3 / 32))


def test_quant_min_bits_is_the_ceiling_of_log2_levels():
    q = C.QuantCapacity()
    for rib, d in ((6.0, 3), (3.3, 32), (10.0, 5)):
        assert q.min_bits(rib, d) == math.ceil(math.log2(q.min_levels(rib, d)))


def test_quant_degenerates_to_one_bit_at_typical_scale():
    """Documented behavior: at MNIST/CIFAR scale (large d, small information) q_min collapses to ~1 bit,
    which is exactly why the quantization screen is only informative when d is small."""
    q = C.QuantCapacity()
    assert q.min_bits_per_dim(3.3, 32) < 0.2
    assert q.min_bits(3.3, 32) == 1
    assert q.min_bits(6.0, 3) == 2                                 # 2^(6/3) = 4 levels -> exactly 2 bits
    assert q.min_bits(6.5, 3) == 3                                 # the ceiling bites in between


def test_distill_min_codebook_size_is_ceil_exp_rib():
    d = C.DistillCapacity()
    assert d.min_codebook_size(4.16) == 65                        # ceil(e^4.16) = ceil(64.07)
    assert d.min_codebook_size(2.0) == 8                          # ceil(e^2) = ceil(7.39)
    assert d.min_codebook_size(0.0) == 2                          # documented floor: a codebook needs >= 2
    assert d.min_codebook_size(4.20) >= d.min_codebook_size(4.16)  # monotone in the requirement


def test_distill_min_capacity_nats_is_the_requirement_passthrough():
    d = C.DistillCapacity()
    for R in (0.0, 1.36, 4.16):
        assert d.min_capacity_nats(R) == approx(R)                 # documented: the bound IS R_ib
    assert d.min_codebook_size(4.16) == math.ceil(math.exp(4.16))


def test_task_info_fano_formula_and_limits():
    assert C.task_info_fano(0.0, 10) == approx(math.log(10))
    assert C.task_info_fano(0.9, 10) < C.task_info_fano(0.1, 10)   # decreasing in the error rate
    assert C.task_info_fano(0.1, 10) <= math.log(10)
    assert C.task_info_fano(0.1, 10) == approx(1.7578, abs=1e-4)
    assert C.task_info_fano(0.5, 2) >= 0.0


def test_task_info_fano_uses_the_label_entropy_ceiling():
    # the requirement can never exceed ln C: that ceiling is what makes K >= C in-excludable
    for C_n in (2, 10, 1000):
        assert C.task_info_fano(0.001, C_n) <= math.log(C_n) + 1e-9
    # C is clamped to >= 2: the Fano expression needs two classes, and a 1-class task is degenerate
    assert C.task_info_fano(0.0, 1) == approx(math.log(2))
    assert C.task_info_fano(0.5, 1) == approx(0.0)
