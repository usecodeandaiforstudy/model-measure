"""Representation geometry and the wall probe (geometry.py, fisher.py, monitor.py)."""

import math

from support import approx
import torch
import torch.nn as nn

import capacity_lens as C


def _clusters(n_per=80, n_classes=4, dim=8, spread=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_classes, dim, generator=g) * 6.0
    y = torch.arange(n_classes).repeat_interleave(n_per)
    x = centers[y] + spread * torch.randn(len(y), dim, generator=g)
    return x, y


def test_separation_ratio_is_larger_for_separated_clusters():
    x_far, y = _clusters(spread=0.5)
    x_near, _ = _clusters(spread=6.0)
    assert C.separation_ratio(x_far, y) > C.separation_ratio(x_near, y)


def test_silhouette_is_higher_for_separated_clusters():
    x_far, y = _clusters(spread=0.5)
    x_near, _ = _clusters(spread=6.0)
    assert C.silhouette_by_label(x_far, y) > C.silhouette_by_label(x_near, y)


def test_quantization_error_is_zero_on_an_exact_codebook():
    feats = torch.eye(4) * 3.0
    err = C.vq_quantization_error(feats, feats.clone(), metric="euclid")
    assert float(err) == approx(0.0, abs=1e-6)


def test_err_1nn_bounds_and_direction():
    x_far, y = _clusters(spread=0.3)
    x_near, _ = _clusters(spread=9.0)
    e_far = C.err_1nn(x_far, y)
    e_near = C.err_1nn(x_near, y)
    assert 0.0 <= e_far <= 1.0 and 0.0 <= e_near <= 1.0
    assert e_far < e_near


def test_wall_probe_returns_the_documented_fields_and_flips_with_overlap():
    x_clean, y = _clusters(spread=0.2)
    x_overlap, _ = _clusters(spread=12.0)
    clean = C.wall_geometry_probe(x_clean, y)
    overlap = C.wall_geometry_probe(x_overlap, y)
    for probe in (clean, overlap):
        assert {"wall", "exclusion_applicable", "k_star_locatable", "err_1nn"} <= set(probe.keys())
        assert probe["wall"] in {"discrete", "diffuse"}
    # a clean multi-cluster wall must be the locatable kind, an overlapping one must not promise K*
    assert clean["wall"] == "discrete"
    assert clean["k_star_locatable"] is True
    assert overlap["wall"] == "diffuse"
    assert overlap["k_star_locatable"] is False


def test_fisher_effective_dimension_runs_and_is_positive():
    """Usage follows the experiments: the caller does its own backward, then calls collect()."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
    f = C.FisherEffectiveDim(model, max_batches=2)
    for _ in range(3):                                             # fewer than 2 batches -> documented None
        out = model(torch.randn(64, 8))
        ((out - torch.randn(64, 1)) ** 2).mean().backward()
        f.collect()
    eff = f.effective_dim()
    assert eff is not None and float(eff) > 0.0
    f.reset()
    assert f.effective_dim() is None


def test_train_monitor_reports_capacity_in_nats_and_no_overfit_verdict():
    """Regression for the 2026-09-30 change: the unverified `H(Z) > R_IB => overfit` flag was removed, and
    the summary's numbers are nats (the earlier version printed "bits" over natural-log values)."""
    torch.manual_seed(0)
    vq = C.VQBottleneck(8, 16)
    mon = C.TrainMonitor(vq)
    _, idx = vq(torch.randn(500, 8))
    out = mon.step(idx)
    assert set(out.keys()) == {"I_XT", "utilization"}      # no verdict-shaped key
    s = mon.summary()
    assert {"logK", "utilization", "I_XT"} <= set(s.keys())
    assert s["logK"] == approx(math.log(16))               # natural log, i.e. nats
    assert 0.0 <= s["utilization"] <= 1.0
    assert "overfit" not in mon.history
