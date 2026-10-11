"""Training diagnostics: monitor the codebook's information capacity and utilization.

What this module does: reports the codebook entropy H(Z) as a proxy for I(X;T), the codebook utilization,
and the bound tightness H(Z) / log K.

**The overfit criterion was REMOVED on 2026-09-30, not merely marked unverified.** The earlier version
returned an `overfit` flag from `H(Z) > R_IB`. It was dropped for two measured reasons and one structural
one:
  1. measured: H(Z) is a representation-entropy UPPER bound on I(X;T) and systematically OVERESTIMATES the
     task information (5.36 nats vs the task's true 4.16 on the synthetic 64-class task), so the comparison
     fires on the estimator's bias rather than on the model;
  2. measured: the project's CP-H/PA4 campaigns show that capacity-vs-task questions are decided by the
     Fano EXCLUSION direction (`log K < R_IB` => the bottleneck provably cannot carry the task), which is
     sound because R_IB is derived from an ACHIEVED error;
  3. structural: comparing an upper bound against a requirement cannot support a negative verdict in either
     direction -- "capacity exceeds what the task needs" is not implied by "H(Z) > R_IB", and the reverse
     reading is exactly the false-exclusion failure mode the project measured elsewhere.
An unverified verdict-shaped API is worse than no API: a caller cannot tell a heuristic from a rule.
Capacity/utilization monitoring itself remains, with its units stated correctly (nats, matching
`capacity_lens` elsewhere; the earlier summary printed "bits" over natural-log values, which is fixed here).
"""

import math


class TrainMonitor:
    """After inserting a VQ bottleneck, collect information-capacity metrics.

    Usage:
        net.vq = VQBottleneck(d, K)
        monitor = TrainMonitor(net.vq)
        for epoch:
            out, idx = net(x)
            m = monitor.step(idx)                 # {"I_XT": ..., "utilization": ...}
        monitor.summary()
    """

    def __init__(self, bottleneck):
        self.bottleneck = bottleneck
        self.history = {"I_XT": [], "utilization": []}

    def step(self, idx):
        """Record one step (all quantities in nats, the project's unit throughout).

        Args:
            idx: codebook indices of the most recent batch [B] (or any shape; flattened internally)
        Returns:
            dict: {"I_XT": codebook entropy H(Z), "utilization": codebook utilization in [0, 1]}
        """
        i_xt = self.bottleneck.codebook_entropy(idx).item()
        util = self.bottleneck.codebook_utilization(idx)
        self.history["I_XT"].append(i_xt)
        self.history["utilization"].append(util)
        return {"I_XT": i_xt, "utilization": util}

    def summary(self):
        """Print a diagnostic summary (nats, with the bit equivalent for orientation) and return it."""
        log_k = math.log(self.bottleneck.n_codes)
        latest_ix_t = self.history["I_XT"][-1] if self.history["I_XT"] else 0.0
        latest_util = self.history["utilization"][-1] if self.history["utilization"] else 0.0
        print(f"[monitor] capacity upper bound log K = {log_k:.2f} nats ({log_k / math.log(2):.2f} bits)")
        print(f"[monitor] measured representation entropy H(Z) = {latest_ix_t:.2f} nats "
              f"(utilization {latest_util:.1%})")
        print(f"[monitor] bound tightness H(Z) / log K = {latest_ix_t / log_k:.2f}")
        print("[monitor] note: H(Z) is an UPPER bound on I(X;T); for capacity-vs-task judgements use the "
              "Fano exclusion direction (log K < R_IB), not a H(Z) comparison")
        return {"logK": log_k, "I_XT": latest_ix_t, "utilization": latest_util}
