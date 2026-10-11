"""Bottleneck layers (bottleneck.py), including the regressions behind the 2026-09-30 gauss campaign."""

import math

from support import approx
import torch

import capacity_lens as C


def test_vq_forward_shapes_and_index_range():
    vq = C.VQBottleneck(8, 16)
    z, idx = vq(torch.randn(5, 8))
    assert z.shape == (5, 8)
    assert idx.shape == (5,)
    assert int(idx.min()) >= 0 and int(idx.max()) < 16


def test_vq_supports_leading_dimensions():
    vq = C.VQBottleneck(8, 16)
    z, idx = vq(torch.randn(2, 3, 8))              # transformer-style [B, L, d]
    assert z.shape == (2, 3, 8)
    assert idx.numel() == 6                        # indices come back flattened (evaluate() relies on it)


def test_vq_output_is_a_codebook_vector_and_commit_loss_is_nonnegative():
    vq = C.VQBottleneck(4, 6, metric="euclid")
    z, idx = vq(torch.randn(7, 4))
    # the forward output is exactly the nearest codeword
    assert torch.allclose(z, vq.codebook[idx], atol=1e-6)
    assert float(vq.commit_loss()) >= 0.0


def test_vq_straight_through_passes_gradient_to_the_input():
    vq = C.VQBottleneck(4, 8)
    t = torch.randn(3, 4, requires_grad=True)
    z, _ = vq(t)
    z.sum().backward()
    assert t.grad is not None and torch.isfinite(t.grad).all()


def test_vq_entropy_never_exceeds_log_k():
    vq = C.VQBottleneck(8, 32)
    _, idx = vq(torch.randn(2000, 8))
    assert C.entropy_of(idx, 32) <= math.log(32) + 1e-6


def test_gauss_capacity_matches_the_formula():
    for d, sigma in ((64, 0.5), (16, 1.5), (8, 3.0)):
        b = C.GaussBottleneck(d, sigma)
        assert b.capacity() == approx(C.gauss_capacity(d, 1.0 / sigma ** 2))


def test_gauss_noiseless_has_infinite_capacity():
    assert C.GaussBottleneck(8, 0.0).capacity() == float("inf")


def test_gauss_normalizes_to_unit_rms():
    b = C.GaussBottleneck(16, 0.0, normalize="rms")
    _, _ = b(torch.ones(3, 16) * 5.0)             # noiseless: output is the normalized signal
    out, idx = b(torch.ones(3, 16) * 5.0)
    assert idx is None                            # no discrete codes -> keeps the (Z, idx) convention
    assert float(out.pow(2).mean().sqrt()) == approx(1.0, abs=1e-5)


def test_gauss_noise_is_active_at_eval_time():
    """Regression test for the 2026-09-30 fix: the earlier version applied noise only while training, so
    at evaluation it was not a channel at all and no capacity could be stated for it."""
    torch.manual_seed(0)
    b = C.GaussBottleneck(8, 0.5)
    b.eval()
    t = torch.ones(4, 8)
    a, _ = b(t)
    c, _ = b(t)
    assert not torch.allclose(a, c), "the noise must be part of the channel at eval time"
    assert torch.allclose(a, t, atol=5.0)         # but it still has to be noise, not garbage


def test_gauss_noiseless_is_deterministic_at_eval():
    b = C.GaussBottleneck(8, 0.0)
    b.eval()
    t = torch.randn(4, 8)
    a, _ = b(t)
    c, _ = b(t)
    assert torch.allclose(a, c)


def test_gauss_per_dim_mode_tracks_the_per_dimension_rms():
    torch.manual_seed(0)
    b = C.GaussBottleneck(4, 0.0, normalize="per_dim")
    b.train()
    t = torch.randn(512, 4) * torch.tensor([3.0, 1.0, 0.5, 0.1])
    for _ in range(400):
        b(t)
    # the running estimate must move toward the input's per-dimension RMS (3.0 / 1.0 / 0.5 / 0.1)
    assert b.dim_rms[0] > b.dim_rms[3]
    assert float(b.dim_rms[0]) > 1.5
