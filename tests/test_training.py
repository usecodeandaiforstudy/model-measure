"""Training/evaluation glue (training.py): entropy, mutual information, Fano, and one end-to-end smoke test."""

import math

from support import approx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import capacity_lens as C


def test_entropy_of_uniform_and_one_hot():
    assert C.entropy_of(torch.arange(8).repeat(100), 8) == approx(math.log(8), abs=1e-6)
    assert C.entropy_of(torch.zeros(100, dtype=torch.long), 8) == approx(0.0, abs=1e-6)


def test_mutual_info_is_zero_for_independent_labels():
    torch.manual_seed(0)
    idx = torch.randint(0, 8, (4000,))
    y = torch.randint(0, 4, (4000,))
    assert abs(C.mutual_info_of(idx, y, 8, 4)) < 0.05


def test_mutual_info_is_the_label_entropy_when_perfectly_predictive():
    idx = torch.arange(4000) % 4
    y = idx.clone()
    # the plug-in carries a (nx-1)(ny-1)/(2n) bias correction: 9/8000 = 0.0011 nats here
    assert C.mutual_info_of(idx, y, 4, 4) == approx(math.log(4), abs=5e-3)


def test_joint_code_is_injective_on_the_state_tuple():
    a = torch.tensor([0, 1, 0, 1])
    b = torch.tensor([0, 0, 1, 1])
    code = C.joint_code([a, b], [2, 2])
    assert len(torch.unique(code)) == 4
    assert int(code.max()) < 4


class _TinyVQNet(nn.Module):
    def __init__(self, d=8, k=16, n_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(16, d)
        self.vq = C.VQBottleneck(d, k)
        self.fc2 = nn.Linear(d, n_classes)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        z, idx = self.vq(h)
        return self.fc2(z), [idx]


def _blobs(n_per_class=60, n_classes=4, dim=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_classes, dim, generator=g) * 4.0
    y = torch.arange(n_classes).repeat_interleave(n_per_class)
    x = centers[y] + torch.randn(len(y), dim, generator=g)
    return TensorDataset(x, y)


def test_train_and_evaluate_end_to_end_smoke():
    torch.manual_seed(0)
    ds = _blobs()
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    model = _TinyVQNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(12):
        C.train_one_epoch(model, loader, opt, device="cpu")
    acc, idx_lists = C.evaluate(model, loader, device="cpu")
    assert acc >= 0.85, "the loop must actually train a separable 4-class task"
    assert idx_lists is not None and len(idx_lists) == 1
    # the monitored quantity is bounded by the codebook
    assert C.entropy_of(idx_lists[0], 16) <= math.log(16) + 1e-6


def test_evaluate_returns_none_indices_without_a_vq():
    class Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 4)

        def forward(self, x):
            return self.fc(x), []

    acc, idx_lists = C.evaluate(Plain(), DataLoader(_blobs(), batch_size=64), device="cpu")
    assert 0.0 <= acc <= 1.0
    assert idx_lists == [] or idx_lists is None


def test_gauss_bottleneck_runs_inside_the_package_loop():
    """A model whose bottleneck is the Gaussian channel must train through the same glue layer (the
    forward returns (Z, None), which evaluate() has to tolerate)."""
    class GaussianNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 8)
            self.vq = C.GaussBottleneck(8, 0.0)
            self.fc2 = nn.Linear(8, 4)

        def forward(self, x):
            z, idx = self.vq(F.relu(self.fc1(x)))
            return self.fc2(z), [idx]

    torch.manual_seed(0)
    loader = DataLoader(_blobs(), batch_size=32, shuffle=True)
    model = GaussianNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(8):
        C.train_one_epoch(model, loader, opt, device="cpu")
    acc, _ = C.evaluate(model, loader, device="cpu")
    assert acc >= 0.8
