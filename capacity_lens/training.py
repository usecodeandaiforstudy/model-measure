"""Minimal training/evaluation loop: automatically handles the VQ commitment loss and (optional) distillation.

Designed as a "batteries-included" glue layer -- users can use their own training loop (just manually call
each VQBottleneck's commit_loss), or use these two functions directly for quick integration.
"""

import math

import torch
import torch.nn.functional as F

from .bottleneck import VQBottleneck


def train_one_epoch(model, loader, optimizer, beta=0.25, device="cpu",
                    distill_from=None, T_kd=3.0, restart_dead_codes=True):
    """Train for one epoch.

    Automatically applies the commitment loss to all VQBottlenecks in the model (also works with multiple bottlenecks);
    performs a dead-code reset after each optimizer.step() (the standard VQ-VAE stabilization mechanism, see
    VQBottleneck.update_usage_and_restart), otherwise codebook utilization collapses,
    H(Z) ≪ log K, and the capacity upper bound loses measurability.

    distill_from: teacher model -> distillation objective = hard-label CE + soft-label KL(T=T_kd).
    (Using only soft labels underfits on a weak teacher; standard KD adds the two.)
    """
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out, _ = model(x)
        ce = F.cross_entropy(out, y)
        if distill_from is not None:
            with torch.no_grad():
                t_out, _ = distill_from(x)
            soft = F.kl_div(F.log_softmax(out / T_kd, dim=1),
                            F.softmax(t_out / T_kd, dim=1),
                            reduction="batchmean") * (T_kd * T_kd)
            loss = ce + soft
        else:
            loss = ce
        for m in model.modules():
            if isinstance(m, VQBottleneck):
                loss = loss + beta * m.commit_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if restart_dead_codes:
            for m in model.modules():
                if isinstance(m, VQBottleneck):
                    t = m.last_input()
                    if t is not None:
                        m.update_usage_and_restart(t)
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    """Returns (acc, idx_lists): idx_lists[k] is the full-test-set indices of the k-th VQ bottleneck (flattened to 1D).

    idx_lists is None when the model has no VQ bottleneck, and entries for a bottleneck that reports no
    discrete codes (a continuous channel such as GaussBottleneck returns `(Z, None)`) are empty tensors --
    a model written as `return self.fc2(z), [idx]` therefore works with either bottleneck type.
    The shape of the bottleneck's output indices may be [B] (MLP) or [B, L] (a Transformer's per-token bottleneck);
    here they are uniformly flattened to 1D, so that the downstream entropy_of / bincount work for both architectures.
    """
    model.eval()
    correct = total = 0
    idx_lists = None
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out, info = model(x)
        correct += (out.argmax(dim=1) == y).sum().item()
        total += y.size(0)
        if info is not None:
            if idx_lists is None:
                idx_lists = [[] for _ in info]
            for k, idx in enumerate(info):
                if idx is None:                  # continuous bottleneck: nothing discrete to record
                    continue
                idx_lists[k].append(idx.reshape(-1).cpu())
    if idx_lists is not None:
        idx_lists = [torch.cat(lst) if lst else torch.empty(0, dtype=torch.long) for lst in idx_lists]
    return correct / total, idx_lists


@torch.no_grad()
def entropy_of(idx, n_states):
    """Entropy of a discrete variable H = -Σ p log p (natural logarithm, nats)."""
    counts = torch.bincount(idx, minlength=n_states).float()
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return -(p * p.log()).sum().item()


def task_info_fano(err, n_classes, h_y=None):
    """Task-information lower bound given by the classification error rate (Fano's inequality), in nats.

    I(Z;Y) ≥ H(Y) − [H_b(P_e) + P_e·ln(C−1)]

    Why use it instead of a plug-in mutual-information estimate: when the sample count N ≪ K·C (codebook × classes),
    the plug-in mutual information is severely biased upward and the Miller–Madow correction severely biased downward (measured at N=1920, K=256,
    C=64 it was corrected to 0). The Fano lower bound uses only the error rate, has no binning bias, and for the purpose of "giving a capacity lower bound"
    it errs in the conservative (lower-bound) direction, which is exactly the direction we need.

    Args:
        err: error rate 1−accuracy
        n_classes: number of classes C
        h_y: H(Y) (nats); if not provided, takes ln C assuming a uniform distribution
    Returns:
        lower bound on I(Z;Y) (nats, non-negative)
    """
    C = max(int(n_classes), 2)
    pe = min(max(float(err), 0.0), 1.0 - 1e-12)
    # convention 0*log(0)=0: when pe=0 (perfect classifier) h_b=0, and the R_IB lower bound = H(Y)
    h_b = 0.0 if pe == 0.0 else -(pe * math.log(pe) + (1.0 - pe) * math.log(1.0 - pe))
    h_y_given_z = h_b + pe * math.log(C - 1)
    h_y = math.log(C) if h_y is None else h_y
    return max(h_y - h_y_given_z, 0.0)


@torch.no_grad()
def mutual_info_of(idx, y, k_states, n_classes):
    """Plug-in estimate of the mutual information I(Z;Y) between discrete variables (nats), with Miller–Madow bias correction.

    **Domain-of-applicability warning**: trustworthy only when the sample count N ≫ K·C. When N is of the same order as K·C the plug-in is severely biased upward
    and the correction severely biased downward (measured at N=1920, K=256, C=64 it was corrected to 0). In that case use
    `task_info_fano` instead (the lower bound given by the error rate).

    Args:
        idx: codeword indices [N]
        y: labels [N]
        k_states: codebook size K
        n_classes: number of classes
    Returns:
        I(Z;Y) (nats, bias-corrected and truncated to non-negative)
    """
    idx = idx.long().flatten()
    y = y.long().flatten()
    joint = torch.zeros(k_states, n_classes, dtype=torch.float64)
    joint.index_put_((idx, y), torch.ones(idx.numel(), dtype=torch.float64), accumulate=True)
    n = joint.sum()
    if n <= 0:
        return 0.0
    pxy = joint / n
    px = pxy.sum(dim=1, keepdim=True)
    py = pxy.sum(dim=0, keepdim=True)
    mask = pxy > 0
    denom = px * py
    mi = (pxy[mask] * (pxy[mask] / denom[mask]).log()).sum().item()
    # Miller–Madow correction: subtract (|X_obs|-1)(|Y_obs|-1)/(2N)
    nx = int((px > 0).sum().item())
    ny = int((py > 0).sum().item())
    mi -= (nx - 1) * (ny - 1) / (2.0 * n.item())
    return max(mi, 0.0)


def joint_code(idx_lists, n_codes_list):
    """Multiple bottleneck indices -> joint encoding (integer mixed radix)."""
    code = torch.zeros_like(idx_lists[0])
    for idx, k in zip(idx_lists, n_codes_list):
        code = code * k + idx
    return code
