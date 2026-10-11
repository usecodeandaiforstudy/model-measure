"""Insertable bottleneck layers: discrete (VQ) and stochastic (Gaussian).

Capacity upper bounds (see capacity.py):
- Discrete bottleneck: I(X;Z) <= H(Z) <= log(n_codes); measured on grid64 the wall sits at 1.15x R_IB, so
  log K is a nearly achievable figure.
- Gaussian bottleneck: I(X;Z) <= (d/2) log(1 + Var(T)/sigma^2); with the unit-RMS normalization (mean
  per-dimension variance ~1, so the mean eigenvalue of Cov(T) is ~1 and Jensen gives the tight form) this
  is gauss_capacity(d, 1/sigma^2) and it is a rigorous upper bound. gauss_demo (2026-09-30) verified the
  whole exclusion pipeline on this channel (3/3 seeds) AND measured the usability caveat: on grid64 the
  noisy channel needed ~24 nats for 90% retention where the discrete codebook needed 4.56 -- a 5.3x
  nat-efficiency gap, i.e. this bound overestimates what a trained encoder actually uses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .capacity import gauss_capacity


class VQBottleneck(nn.Module):
    """Learnable codebook bottleneck: nearest-neighbor quantizes the representation T onto a codebook.

    Args:
        d: representation dimension (dimension of the codebook vectors)
        n_codes: number of codebook vectors K. Capacity upper bound = log K.
        metric: "cosine" (default, insensitive to magnitude, suited to representation space)
                or "euclid" (standard VQ-VAE distance, suited to spaces with clear cluster
                structure, e.g. grid-type synthetic tasks -- cosine cannot distinguish
                collinear cluster centers).
    """

    def __init__(self, d, n_codes, metric="cosine", restart=True):
        super().__init__()
        self.d = d
        self.n_codes = n_codes
        self.metric = metric
        self.restart = restart
        self.codebook = nn.Parameter(torch.randn(n_codes, d))
        self._cache = None  # (T, idx) most recent forward, used by commit_loss/restart
        # EMA usage counts (basis for dead-code diagnosis and reset) -- non-parametric, not passed to the optimizer
        self.register_buffer("ema_count", torch.zeros(n_codes))
        self.register_buffer("restart_events", torch.zeros(1))

    def _nearest(self, T):
        """Return the nearest codeword indices [N] (T must already be 2-D [N, d])."""
        if self.metric == "euclid":
            return torch.cdist(T, self.codebook).argmin(dim=-1)
        # cosine nearest neighbor (dot product after normalization), more stable than Euclidean distance
        T_n = F.normalize(T, dim=-1)
        C_n = F.normalize(self.codebook, dim=-1)
        return (T_n @ C_n.t()).argmax(dim=-1)

    def forward(self, T):
        """T: [..., d] -> (Z: [..., d], idx: [...]). Z keeps gradients via straight-through.

        Supports arbitrary leading dimensions (both MLP [B, d] and Transformer [B, L, d]):
        internally flattens to [-1, d] for nearest neighbor, then restores the shape.
        """
        shape = T.shape
        flat = T.reshape(-1, self.d)
        idx = self._nearest(flat)                    # [N]
        Z = self.codebook[idx].reshape(shape)        # [..., d] discrete output
        # straight-through estimator: forward takes the discrete codebook, backward passes gradients to T
        Z = T + (Z - T).detach()
        self._cache = (flat, idx)
        return Z, idx

    def commit_loss(self):
        """commitment loss: pull the codebook vectors toward the representation (gradients flow only to the codebook, not to the representation)."""
        if self._cache is None:
            return self.codebook.new_zeros(())
        flat, idx = self._cache
        return F.mse_loss(self.codebook[idx], flat.detach())

    def last_input(self):
        """Input representation T of the most recent forward (already flattened to [N, d], used for dead-code reset)."""
        return None if self._cache is None else self._cache[0]

    @torch.no_grad()
    def update_usage_and_restart(self, T, decay=0.99, threshold=0.1):
        """EMA-tracks codeword usage and resets long-unused codewords to random representations from the current batch.

        The standard stabilization mechanism of VQ-VAE (dead-code restart of van den Oord et al. 2017):
        under naive VQ dead codes are never selected, utilization collapses, H(Z) ≪ log K, and the
        capacity upper bound cannot reveal its tightness.

        threshold is a ratio relative to the average usage level: usage = ema_count/total*K, where 1.0 is average;
        the default 0.1 means a code is declared dead only when "its usage is below 10% of the average level".

        Returns the number of codewords reset this time (int).
        """
        if not self.restart:
            return 0
        T = T.reshape(-1, self.d)          # compatible with arbitrary leading dimensions
        B = T.shape[0]
        idx = self._nearest(T)
        onehot = F.one_hot(idx, self.n_codes).float()
        self.ema_count.mul_(decay).add_(onehot.sum(dim=0), alpha=1.0 - decay)
        total = self.ema_count.sum()
        if total <= 0:
            return 0
        usage = self.ema_count / total * self.n_codes        # 1.0 = average usage level
        dead = usage < threshold
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return 0
        n_dead = min(n_dead, B)
        dead_idx = torch.nonzero(dead, as_tuple=True)[0][:n_dead]
        src = T[torch.randperm(B, device=T.device)[:n_dead]]
        self.codebook.data[dead_idx] = src.to(self.codebook.dtype)
        # reset this codeword's count to the threshold level, to avoid it being declared dead again immediately next round
        self.ema_count[dead_idx] = threshold * total / self.n_codes
        self.restart_events += n_dead
        return n_dead

    @torch.no_grad()
    def codebook_entropy(self, idx):
        """Codebook usage entropy H(Z) = -sum_k p_k log p_k, an unbiased estimate of the upper bound on I(X;Z)."""
        counts = torch.bincount(idx, minlength=self.n_codes).float()
        total = counts.sum()
        if total == 0:
            return torch.tensor(0.0)
        p = counts / total
        p = p[p > 0]
        return -(p * torch.log(p)).sum()

    @torch.no_grad()
    def codebook_utilization(self, idx):
        """Codebook utilization = number of codewords used / K. Lower means more severe collapse."""
        counts = torch.bincount(idx, minlength=self.n_codes)
        used = (counts > 0).sum().item()
        return used / self.n_codes


class GaussBottleneck(nn.Module):
    """Additive-white-Gaussian-noise channel bottleneck: the CONTINUOUS capacity knob (nats, not codes).

    A channel, not a regularizer. Three things make that true, and all three were missing before the
    gauss_demo campaign (2026-09-30) -- the earlier version added noise only while training and left the
    signal scale uncontrolled, so at evaluation it was not a channel at all and no capacity could be
    stated for it:
      1. the representation is normalized to unit root-mean-square, so the signal variance is CONTROLLED
         instead of inherited from the encoder's scale (no normalization => the nominal capacity below is
         undefined, since it depends on Var(T));
      2. the noise is active at TRAIN AND EVAL time -- it is the channel, so switching it off at eval
         would measure a different (noiseless) model;
      3. forward returns the project's (Z, idx) convention (idx is None: there are no discrete codes),
         so the bottleneck drops into any model built for VQBottleneck.
    Nominal capacity (Shannon, an UPPER bound on what the decoder can receive -- hence sound for
    exclusion judgements, and the continuous analogue of log(n_codes)):
        C_nom = gauss_capacity(d, snr) = (d/2) * ln(1 + 1/sigma^2)      [nats]
    valid while every dimension of the normalized signal has variance <= 1 (checked by the experiment).

    Noise regularization (train-time-only noise, no normalization) is a different tool; use
    sigma=0.0/None plus dropout for that instead of this class.

    Args:
        d: width of the representation (needed for the capacity bookkeeping)
        sigma: noise standard deviation; 0.0 means noiseless (C_nom = inf -- the ceiling arm)
        normalize: "rms" (default) normalizes each vector to unit root-mean-square: it fixes the AVERAGE
            per-dimension variance, so a single dimension may still exceed 1 and the nominal capacity is
            then only a reference number, not a rigorous bound (measured max/mean ratios of 2-8 in
            gauss_demo). "per_dim" keeps a running per-dimension RMS (momentum, BatchNorm-style: updated
            while training, frozen at eval), which makes every dimension's variance <= 1 by construction
            and therefore makes C_nom a rigorous upper bound -- use it when the capacity figure has to
            carry a certified claim. "none" disables normalization (Var(T) then sets the scale, so no
            capacity can be stated).
    """

    def __init__(self, d, sigma=0.5, normalize="rms"):
        super().__init__()
        self.d = d
        self.sigma = float(sigma)
        self.normalize = normalize
        self.momentum = 0.99
        self.register_buffer("dim_rms", torch.ones(d))

    def capacity(self):
        """Nominal capacity in nats (the continuous analogue of log K). inf when sigma = 0."""
        if self.sigma <= 0.0:
            return float("inf")
        return gauss_capacity(self.d, 1.0 / (self.sigma ** 2))

    def forward(self, T):
        """T: [..., d] -> (Z: [..., d], None). The returned idx is always None (no discrete codes)."""
        Z = T
        if self.normalize == "rms":
            rms = Z.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
            Z = Z / rms
        elif self.normalize == "per_dim":
            if self.training:
                with torch.no_grad():
                    cur = Z.reshape(-1, self.d).pow(2).mean(dim=0).sqrt().clamp_min(1e-6)
                    self.dim_rms.mul_(self.momentum).add_(cur, alpha=1.0 - self.momentum)
            Z = Z / self.dim_rms.clamp_min(1e-6)
        if self.sigma > 0.0:
            Z = Z + self.sigma * torch.randn_like(Z)
        return Z, None
