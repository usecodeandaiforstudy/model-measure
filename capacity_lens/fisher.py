"""Fisher effective-dimension estimation (v1 module): participation ratio of the gradient Gram matrix.

Principle: the covariance of SGD mini-batch gradients g_b is proportional to the Fisher information matrix (a standard result of natural-gradient theory).
Collect gradient vectors from several batches; the "effective dimension" of the subspace they span is measured by the participation ratio:

    d_eff = (Σ_i λ_i)^2 / Σ_i λ_i^2

where λ_i are the eigenvalues of the gradient Gram matrix.

- d_eff = 1: all batch gradients are collinear (only one direction is being trained);
- d_eff = B: the B batch gradients are pairwise orthogonal (all directions are active).

Complementary relationship with the capacity lens: codebook entropy H(Z) measures "how much information the representation stores" (forward),
while d_eff measures "in how many directions the parameters are being pushed" (backward).

Computational cost: each step caches only one flattened gradient vector (O(P) memory, P = number of parameters);
effective_dim() does one SVD on the [B, P] matrix, computing only B singular values (O(B^2 P) when B << P).

Fisher-Rao arc length L_info (v2, mechanical form): the information length of the training trajectory

    L_info = Σ_t ds_t,  ds_t = sqrt(Δθ_t^T F̂_t Δθ_t)

where Δθ_t is the parameter displacement at step t, and F̂ is the empirical Fisher from the same source as d_eff (the second moment of the
most recent B mini-batch gradients). The theoretical basis is the Jeffreys local expansion D_KL[p_θ ‖ p_{θ+Δθ}] = ½Δθ^T g Δθ
+ O(Δθ³): the per-step contribution to the information length is ds = sqrt(2·KL_local), whose metric is exactly the Fisher.
See the docstring of FisherArcLength.
"""

import torch


class FisherEffectiveDim:
    """Fisher effective-dimension monitor.

    Usage:
        fed = FisherEffectiveDim(model, max_batches=32)
        for epoch in range(epochs):
            for x, y in loader:
                loss = ...
                loss.backward()
                fed.collect()              # capture the current gradient (before step)
                optimizer.step()
            d_eff = fed.effective_dim()    # computed once per epoch
            fed.reset()                    # recollect every epoch
    """

    def __init__(self, model, max_batches=32):
        """
        Args:
            model: the monitored model (reads the .grad of its .parameters())
            max_batches: maximum number of cached batches B (also the upper bound of d_eff)
        """
        self.model = model
        self.max_batches = max_batches
        self._grads = []
        self.history = []  # record of each effective_dim() call

    @torch.no_grad()
    def collect(self):
        """Call after loss.backward() and before optimizer.step(): cache the current gradient vector."""
        g = torch.cat([p.grad.detach().reshape(-1)
                       for p in self.model.parameters() if p.grad is not None])
        self._grads.append(g.to(torch.float32).cpu())
        if len(self._grads) > self.max_batches:
            self._grads.pop(0)

    @torch.no_grad()
    def effective_dim(self):
        """Participation ratio d_eff = (Σλ)²/Σλ², where λ are the squared singular values of the [B, P] gradient matrix.

        Returns a float; returns None when fewer than 2 batches are available.
        """
        if len(self._grads) < 2:
            return None
        G = torch.stack(self._grads)                  # [B, P]
        s = torch.linalg.svdvals(G)                   # B singular values σ_i
        lam = s ** 2                                  # Gram eigenvalues λ_i = σ_i²
        denom = (lam ** 2).sum()
        if denom == 0:
            return 0.0                                # all-zero gradients (training stalled)
        d_eff = (lam.sum() ** 2 / denom).item()
        self.history.append(d_eff)
        return d_eff

    def reset(self):
        """Clear the gradient cache (called once per epoch, to measure the subspace within an epoch)."""
        self._grads = []


def arc_length_step(dtheta, grads):
    """Single-step Fisher-Rao information length (pure function, independently unit-testable).

    ds = sqrt( (1/B) · Σ_b (g_b · Δθ)² )

    grads is the [B, P] gradient matrix (rank-B estimate of the empirical Fisher F̂ = (1/B)Σ g_b g_b^T),
    dtheta is the [P] parameter displacement of this step, so ds² = Δθ^T F̂ Δθ.

    Properties:
    - each g_b·Δθ is the first-order change in the loss of batch b -- a scalar, so ds is
      first-order invariant to reparameterization of the parameters (holds for small step sizes);
    - ds has the same units as the loss (a reading of "first-order work");
    - ds=0 when the single gradient is orthogonal to the displacement: under the Fisher metric, a parameter displacement that "does not go far" contributes no arc length.
    """
    if grads.dim() != 2 or dtheta.dim() != 1:
        raise ValueError("grads must be [B, P], dtheta must be [P]")
    if grads.shape[0] == 0:
        raise ValueError("grads must contain at least one batch")
    if grads.shape[1] != dtheta.shape[0]:
        raise ValueError(f"dimension mismatch: grads P={grads.shape[1]}, dtheta P={dtheta.shape[0]}")
    proj = grads @ dtheta                       # [B] first-order loss change per batch
    return float(torch.sqrt((proj ** 2).mean()))


class FisherArcLength:
    """Fisher-Rao arc-length monitor: L_info = Σ_t sqrt(Δθ_t^T F̂ Δθ_t).

    Mechanical correspondence: after equipping the configuration space with the Fisher metric, the cumulative arc length of the training trajectory
    (a reparameterization-invariant measure of "how far the model actually moved", more honest than the epoch count).
    The per-step contribution ds_t is given by the Jeffreys local expansion D_KL[p_θ‖p_{θ+Δθ}] = ½Δθ^T g Δθ + O(Δθ³)
    as: ds = sqrt(Δθ^T g Δθ). F̂ is estimated using the second moment of the most recent B mini-batch gradients
    (same source as FisherEffectiveDim; old batches are computed at old θ, a monitor-level approximation).

    Usage (no need to modify the training loop, just attach it to the optimizer):

        fa = FisherArcLength(model, max_batches=32)
        fa.attach(optimizer)
        for epoch in range(epochs):
            train_one_epoch(model, loader, optimizer)   # normal training
            print(fa.total)                # cumulative arc length (monotonically increasing)
        rates = fa.history                 # per-step ds history

    Implementation notes:
    - collected via the optimizer's pre/post-step hooks: capture gradients + a parameter snapshot before step,
      compute Δθ after step. The direct parameter jumps from VQ dead-code resets happen after step and are not counted
      (a discrete maintenance operation, not gradient flow).
    - for convergence readings, prefer the **per-epoch increment ΔL_info** over the absolute value: under noisy SGD the
      absolute value keeps growing linearly with the learning rate, and only the increment rate reflects how fast the function actually moves.
    - memory is of the same order as FisherEffectiveDim: B flattened gradient vectors + 1 parameter snapshot.
    """

    def __init__(self, model, max_batches=32):
        """
        Args:
            model: the monitored model (reads the .grad and .data of its .parameters())
            max_batches: number of batches B in the Fisher estimation window
        """
        self.model = model
        self.max_batches = max_batches
        self._grads = []      # most recent B mini-batch gradients (oldest first)
        self._theta = None    # parameter snapshot before this step (set to None after consumption)
        self._params = None   # parameter list at snapshot time (to ensure consistency across the before/after)
        self.total = 0.0      # cumulative arc length L_info
        self.history = []     # per-step ds

    def attach(self, optimizer):
        """Register pre/post-step hooks on the optimizer (torch >= 2.0), returns self."""

        def pre_hook(opt, args, kwargs):
            with torch.no_grad():
                ps = [p for p in self.model.parameters() if p.grad is not None]
                if not ps:
                    return
                self._params = ps
                g = torch.cat([p.grad.detach().reshape(-1) for p in ps])
                self._grads.append(g.to(torch.float32).cpu())
                if len(self._grads) > self.max_batches:
                    self._grads.pop(0)
                self._theta = torch.cat(
                    [p.detach().reshape(-1) for p in ps]).to(torch.float32).cpu()

        def post_hook(opt, args, kwargs):
            if self._theta is None or not self._params:
                return
            theta_new = torch.cat(
                [p.detach().reshape(-1) for p in self._params]
            ).to(torch.float32).cpu()
            dtheta = theta_new - self._theta
            # only gradients in the window whose length matches this step's parameter set participate (guards against crashes when the parameter set unexpectedly changes)
            grads = [g for g in self._grads if g.numel() == dtheta.numel()]
            if grads:
                ds = arc_length_step(dtheta, torch.stack(grads))
                self.total += ds
                self.history.append(ds)
            self._theta = None  # consumed

        optimizer.register_step_pre_hook(pre_hook)
        optimizer.register_step_post_hook(post_hook)
        return self
