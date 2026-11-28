"""Integration tests: the five modules driven by REAL torch training, not fake callbacks.

This is the gap flagged in reference/notes/pkg-buildout.md: every unit test above injects a fake callback, so
nothing proved the modules work against an actual model. Here each module is exercised once end to end on a
tiny deterministic model. Assertions are structural plus a determinism check (the same call twice must agree),
because real training is not bit-identical across machines; the exact-number checks live in the unit tests.

Kept deliberately small: 200 samples, 2 epochs, one seed => a few seconds.
"""
from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from capacity_lens.bit_curves import activation_curve, pruning_curve
from capacity_lens.cost import certify_lower, constraint_cost, is_constrained_local_min
from capacity_lens.capacity_probe import measure_capacity
from capacity_lens.represent import align_report
from capacity_lens.tradeoff import codec_tradeoff
from capacity_lens.batch_delta import (delta_summary, online_step_delta, per_batch_delta,
                                       select_batches, track_probe)

D_IN, H, C, N = 8, 6, 4, 200


def _make(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, C, (n,), generator=g)
    mu = torch.zeros(C, D_IN, dtype=torch.float64)
    for j in range(C):
        mu[j, j % D_IN] = 2.0
    return mu[y] + 0.8 * torch.randn(n, D_IN, generator=g, dtype=torch.float64), y


def _model():
    return nn.Sequential(nn.Linear(D_IN, H, dtype=torch.float64), nn.Tanh(),
                         nn.Linear(H, C, dtype=torch.float64))


def _loss(m, x, y):
    return F.cross_entropy(m(x), y)


def _hvp(m, x, y):
    def f(v):
        params = [p for p in m.parameters()]
        outs = []
        grads = torch.autograd.grad(_loss(m, x, y), params, create_graph=True)
        flat = torch.cat([g.reshape(-1) for g in grads])
        dot = (flat * v).sum()
        second = torch.autograd.grad(dot, params, retain_graph=False)
        outs.extend([s.reshape(-1) for s in second])
        return torch.cat(outs)
    return f


def _train(m, x, y, steps=60, lr=0.05):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad(); _loss(m, x, y).backward(); opt.step()
    return m


def test_cost_on_real_model():
    x, y = _make(N, 1)
    m = _train(_model(), x, y)
    P = sum(p.numel() for p in m.parameters())
    # a 4-dimensional subspace
    B = torch.linalg.qr(torch.randn(P, 4, generator=torch.Generator().manual_seed(7),
                                    dtype=torch.float64)).Q[:, :4]
    params = [p for p in m.parameters()]
    g = torch.cat([t.reshape(-1) for t in torch.autograd.grad(_loss(m, x, y), params)])
    g_perp = g - B @ (B.T @ g)
    rep = constraint_cost(g_perp, _hvp(m, x, y))
    assert rep["unit"] == "loss" and rep["certified"] is False
    assert isinstance(rep["value"], float) and rep["value"] >= 0.0
    lo = certify_lower(g_perp, _hvp(m, x, y))
    assert lo["certified"] is True and lo["lower"] <= rep["value"] + 1e-9


def test_restricted_min_on_real_model():
    x, y = _make(N, 2)
    m = _train(_model(), x, y)
    P = sum(p.numel() for p in m.parameters())
    B = torch.eye(P, dtype=torch.float64)
    r = is_constrained_local_min(_hvp(m, x, y), B, n_grid=8)
    assert isinstance(r["satisfied"], bool) and isinstance(r["lambda_min_restricted"], float)


def test_capacity_with_real_fitfn():
    calls = {"n": 0}

    def fit_fn(n, steps, seed):
        calls["n"] += 1
        x, y = _make(n, 100 + seed)
        m = _train(_model(), x, y, steps=max(10, steps // 10))
        with torch.no_grad():
            err = float((m(x).argmax(1) != y).float().mean())
        return err < 0.5

    rep = measure_capacity(fit_fn, budget=10, draws=5, grid=(50, 100, 200), arch={"d_in": D_IN})
    assert isinstance(rep["capacity"], float) and isinstance(rep["saturated"], bool)
    assert rep["rules"]["draws"] == 5 and rep["rules"]["architecture_reported"] is True
    assert calls["n"] > 0


def test_capacity_is_repeatable_with_real_training():
    def make_fit():
        def fit_fn(n, steps, seed):
            x, y = _make(n, 200 + seed)
            m = _train(_model(), x, y, steps=max(10, steps // 10))
            with torch.no_grad():
                return float((m(x).argmax(1) != y).float().mean()) < 0.5
        return fit_fn

    a = measure_capacity(make_fit(), budget=10, draws=5, grid=(50, 100))
    b = measure_capacity(make_fit(), budget=10, draws=5, grid=(50, 100))
    assert a["capacity"] == b["capacity"] and a["saturated"] == b["saturated"]


def test_represent_with_real_gradients():
    x, y = _make(N, 3)
    m = _train(_model(), x, y)
    params = [p for p in m.parameters()]

    def gn(xx, yy):
        return float(torch.cat([t.reshape(-1) for t in
                                torch.autograd.grad(_loss(m, xx, yy), params)]).norm())

    base = [gn(x, y) for _ in range(12)]
    rep = align_report({"same": gn(x, y), "double": 2.0 * gn(x, y)}, base)
    names = {s["name"]: s["verdict"] for s in rep["subsets"]}
    assert names["same"] == "representative"
    assert rep["validity"]["window_steps"] == 100


def test_bits_with_real_quantisation():
    x, y = _make(N, 4)
    m = _train(_model(), x, y)

    def loss_at(bits):
        mq = _model(); mq.load_state_dict(m.state_dict())
        with torch.no_grad():
            for p in mq.parameters():
                lo, hi = float(p.min()), float(p.max())
                lv = 2 ** bits
                step = (hi - lo) / max(lv - 1, 1)
                p.copy_(torch.round((p - lo) / max(step, 1e-12)).clamp(0, lv - 1) * step + lo)
            return float(_loss(mq, x, y))

    base = float(_loss(m, x, y).detach())
    r = activation_curve({1: loss_at(1), 2: loss_at(2), 4: loss_at(4), 8: loss_at(8)}, base)
    assert r["units"] == "loss" and set(r["delta"]) == {1, 2, 4, 8}


def test_pruning_curve_with_real_model():
    x, y = _make(N, 5)
    m = _train(_model(), x, y)
    base = float(_loss(m, x, y).detach())

    def loss_with_zeros(k):
        mq = _model(); mq.load_state_dict(m.state_dict())
        with torch.no_grad():
            flat = torch.cat([p.reshape(-1) for p in mq.parameters()])
            idx = flat.abs().argsort()[:k]
            off = 0
            for p in mq.parameters():
                sel = idx[(idx >= off) & (idx < off + p.numel())] - off
                p.view(-1)[sel] = 0.0
                off += p.numel()
            return float(_loss(mq, x, y))

    r = pruning_curve({k: loss_with_zeros(k) for k in (1, 4, 16, 64)}, base)
    assert r["units"] == "loss" and isinstance(r["delta"], dict)


def test_tradeoff_with_real_pca():
    x, _ = _make(N, 6)
    mu = x.mean(0, keepdim=True)
    U, S, V = torch.linalg.svd(x - mu, full_matrices=False)
    pca_curve = {}
    ae_curve = {}
    for k in (2, 4):
        Vk = V[:k]
        rec = ((x - mu) @ Vk.T @ Vk + mu).clamp(-3, 3)
        pca_curve[k] = float(F.mse_loss(rec, x))
        ae_curve[k] = pca_curve[k] * 1.1          # a deliberately worse synthetic 'autoencoder'
    r = codec_tradeoff(pca_curve, ae_curve)
    assert r["winner_at"][2] == "pca" and r["crossover_k"] == 2


def test_batch_delta_on_a_real_model():
    x, y = _make(N, 8)
    m = _train(_model(), x, y)
    ps = [p for p in m.parameters()]
    g_p = torch.cat([t.reshape(-1) for t in torch.autograd.grad(_loss(m, x, y), ps)])
    full = float(_loss(m, x, y).detach())
    per_batch = {}
    for b in range(4):
        xb, yb = x[b * 25:(b + 1) * 25], y[b * 25:(b + 1) * 25]
        g_b = torch.cat([t.reshape(-1) for t in torch.autograd.grad(_loss(m, xb, yb), ps)])
        per_batch[b] = online_step_delta(float((g_p * g_b).sum()), lr=1e-3)["delta_step"] + full
    d = per_batch_delta(per_batch, full)
    s = delta_summary(d)
    assert s["n"] == 4 and isinstance(s["cv"], float)
    picked = select_batches(d, frac=0.5, mode="high")
    assert len(picked) == 2 and picked == select_batches(d, frac=0.5, mode="high")


def test_track_probe_needs_only_numbers():
    # track_probe needs at least three epochs to report a correlation, so give it three
    r = track_probe([-5e-3, -5e-3, -3e-3, -3e-3, -1e-3, -1e-3], [1.0, 0.9, 0.8])
    assert r["epoch_tracking_spearman"] is not None and r["epoch_tracking_spearman"] < -0.9
    assert r["spread_ratio"] > 0


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
