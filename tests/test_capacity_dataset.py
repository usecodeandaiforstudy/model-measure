"""Tests for the capacity_dataset module (data report + codec attribution).

Same style as the rest of the suite: plain test functions, no pytest required.
"""
import sys
import pathlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import capacity_dataset as CD


def _data(n=600, d=6, c=3, seed=0):
    torch.manual_seed(seed)
    y = torch.randint(0, c, (n,))
    mu = torch.zeros(c, d)
    for j in range(c):
        mu[j, j] = 1.6
    return mu[y] + 0.8 * torch.randn(n, d), y


def test_package_selftest_is_clean():
    assert CD.selftest() == []


def test_dup_exact_rate():
    x, y = _data(600)
    xd = torch.cat([x, x[:100]])
    yd = torch.cat([y, y[:100]])
    out = CD.dup_report(xd, yd)
    assert abs(out["dup_rate_12dp"] - 100 / 700) < 1e-6


def test_dup_collisions_detected():
    x, y = _data(200)
    y2 = (y + 1) % 3
    out = CD.dup_report(torch.cat([x, x]), torch.cat([y, y2]))
    assert out["collisions_12dp"] > 0


def test_single_class_and_tiny_data_do_not_crash():
    x = torch.randn(20, 4)
    y = torch.zeros(20, dtype=torch.long)
    out = CD.build_report(x, y, epochs=2)
    assert out["recommendation"]["tag"] in {"DEDUP FIRST", "TOO EASY", "USABLE", "UNREACHABLE"}


def test_price_requires_truth():
    x, y = _data(300)
    assert CD.review_price(x, None, y) is None


def test_price_refuses_when_no_errors_exist():
    x, y = _data(300)
    out = CD.review_price(x, y, y)
    assert out is None or "unavailable" in out


def test_report_without_truth_has_no_price():
    x, y = _data(300)
    out = CD.build_report(x, y, epochs=3)
    assert out["price"] is None
    assert out["recommendation"]["why"]


def test_recommendation_precedence_dedup_first():
    dup = {"dup_rate_12dp": 0.2}
    sat = {"level": 1.0, "param_knee": 0.05}
    tag, _ = CD.recommend(dup, sat)
    assert tag == "DEDUP FIRST"


def test_determinism_same_seed():
    x, y = _data(500)
    a = CD.saturation_report(x, y, seed=0, epochs=6)
    b = CD.saturation_report(x, y, seed=0, epochs=6)
    assert abs(a["level"] - b["level"]) < 1e-12
    assert a["param_knee"] == b["param_knee"]


def test_unavailable_is_always_a_list():
    x, y = _data(200)
    out = CD.build_report(x, y, epochs=2)
    assert isinstance(out["dup"]["unavailable"], list)
    assert isinstance(out["saturation"]["unavailable"], list)


def test_latent_report_flags_reliability():
    x, y = _data(400, d=3)
    out = CD.latent_report(x, y)
    assert out["latent"] == 3
    assert out["h_z_reliable"] is True
    assert isinstance(out["unavailable"], list)


def test_codec_report_is_undecided_without_a_reference():
    torch.manual_seed(0)
    x = torch.randn(300, 5)
    y = torch.randint(0, 3, (300,))
    enc = torch.nn.Sequential(torch.nn.Linear(5, 3), torch.nn.ReLU())
    dec = torch.nn.Sequential(torch.nn.Linear(3, 3))
    out = CD.codec_report(enc, dec, x, y, reference_acc=None, epochs=3)
    assert out["verdict"].startswith("UNDECIDED")
    assert out["reference_acc"] is None
