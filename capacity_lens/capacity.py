"""Capacity upper-bound computations (pure functions, independent of any training framework).

All are standard information-theoretic results; this project makes no theoretical innovation, only engineering implementation.
"""

import math


def discrete_capacity_logK(n_codes):
    """Discrete bottleneck capacity upper bound log K (natural logarithm).

    Args:
        n_codes: codebook size K (number of distinguishable states)
    Returns:
        log K
    """
    return math.log(n_codes)


def discrete_capacity_uniform(d, q):
    """Uniform quantization: q levels per dimension over d dimensions, K = q^d, capacity = d log q.

    Args:
        d: width (dimension)
        q: number of quantization levels per dimension
    Returns:
        d * log q
    """
    return d * math.log(q)


def gauss_capacity(d, snr):
    """Gaussian bottleneck capacity (d/2) log(1 + snr).

    Args:
        d: width
        snr: signal-to-noise ratio P/sigma^2
    Returns:
        (d/2) * log(1 + snr)

    **Status: VERIFIED end to end on the continuous side (gauss_demo, 2026-09-30, grid64, 3 seeds), with a
    measured usability caveat.** With the unit-RMS normalization of GaussBottleneck the formula with
    snr = 1/sigma^2 is a rigorous UPPER bound (Jensen: I(z;z+n) <= (d/2)ln(1+lambda_bar/sigma^2), and the
    normalization fixes the mean eigenvalue lambda_bar at ~1). Measured: (i) the certified exclusion
    transfers -- every channel whose bound was below R_IB stayed at retention <= 0.32 (C2, 3/3 seeds);
    (ii) the bound was never violated by a measured student (C1, 3/3); (iii) the continuous wall's
    sandwich [R_IB(eps), C_rig] contained the discrete arm's exact wall (C3, 3/3). Caveat: the formula is
    a usable-capacity OVERESTIMATE on this task -- the noisy channel needed ~24 nats to retain 90% of the
    teacher where the discrete codebook needed only log 96 = 4.56 nats (5.3x), and at 4 nats (the
    certified boundary) the trained encoder used only ~0.4 of what the channel could carry.
    See docs/continuous-bottleneck.md.
    """
    return 0.5 * d * math.log1p(snr)


def min_capacity_serial(capacities):
    """Serial bottlenecks: capacity is determined by the narrowest bottleneck (DPI shrinks stage by stage).

    Args:
        capacities: list of individual bottleneck capacities [log K_1, log K_2, ...]
    Returns:
        min(capacities)

    **Applicability boundary (important)**: holds only for **pure serial topologies with no bypass** (already
    verified by the P2 experiment: a 3xlog6 series locked MNIST accuracy at 0.174, whereas a single log256 reached 0.886).
    Once an identity/residual bypass exists, see `sum_capacity_parallel`.
    """
    return min(capacities)


def sum_capacity_parallel(capacities):
    """Parallel/residual branches: capacities can add up (the counterpart of the serial min form).

    For serial (no bypass), DPI shrinks stage by stage -> end-to-end capacity = min_l C_l;
    once an identity/residual bypass exists, information can circumvent the bottleneck and the **min bound fails**:
    here each branch's capacity is counted independently, and the end-to-end capacity is approximately the **sum of the branches** -- an
    identity path, if full precision, is not constrained by the bottleneck at all.

    Args:
        capacities: list of capacities of the parallel branches
    Returns:
        sum(capacities)

    Engineering implication: **the topology must be determined first** before reporting a serial min bound. Applying the
    min form to a network with skip connections severely underestimates capacity (exactly the misuse this project aims to prevent).
    """
    return sum(capacities)


def conditional_capacity(n_experts, per_expert, top_k, shared=0.0):
    """Two-tier capacity bounds for conditional computation (routing x local functions, MoE-style).

    Grounding (framework.md / A-002, 2026-10): routing does not create information, it changes the
    "capacity--compute--storage" exchange rate; the TOTAL parameters of a MoE are REAL capacity (storage),
    while a single example only activates a top-k subset (flow). So the matching question splits into two
    tiers that the serial/parallel rules of this module are degenerate cases of:

      storage = shared + n_experts * per_expert      (can the mapping be stored at all?  sum over experts)
      flow    = shared + top_k * per_expert          (can one example carry the task info through?  min along
                                                      the activated path = the top-k subset's contribution)

    Args:
        n_experts: total number of routed experts
        per_expert: capacity of one expert (e.g. discrete_capacity_logK(K_e))
        top_k: number of experts activated per example
        shared: capacity of the shared (always-activated) part
    Returns:
        dict(storage=..., flow=...) in the same units (nats)

    **Status: VERIFIED (moe_demo, 2026-09-30, route64 task, 6 seeds total).** The two-tier FORMULA is
    confirmed: (i) the storage tier is real capacity (joint codebook utilization 84-94% at the wall
    crossing, T2 passed); (ii) the flow tier (router info + top-k expert info) predicts the routed curve
    when anchored at the task's MEASURED dense wall: F1a 2/2 judgeable seeds predicted ke=32 (flow=ln256)
    and routed reached 1.000/0.999 there; the third seed had no detectable dense wall (its dense K=256
    student stalled at 0.715) yet its routed(32)=0.998 beat its best dense -- consistent, recorded as
    unjudgeable-not-failing; F1b 3/3 (routed >= dense(8*K_e) - 0.05 for every pair -- reduction plus the
    equal-total routing advantage, the A-002 5.3 interference mechanism). The pre-registered first run
    anchored at the Fano bound ln64 FAILED: the Fano bound underestimates the practical wall by 4x on
    this task family (the CP-H-32x-margin family). **The anchor for the flow tier must be the MEASURED
    wall (dense curve), not the Fano bound.** The serial min (`flow` tier) and parallel sum (`storage`
    tier) remain the fully experiment-backed forms (P2/P4); this function's flow tier is verified in
    formula with the measured-wall anchor, single task family (route64).
    """
    if top_k < 0 or top_k > n_experts:
        raise ValueError(f"top_k {top_k} is outside [0, n_experts={n_experts}]")
    return dict(storage=shared + n_experts * per_expert,
                flow=shared + top_k * per_expert,
                note="verified (route64 task family, 6 seeds): storage=T2 passed, flow=anchored at the measured dense wall (do not use the Fano lower bound as the wall anchor)")


def kv_storage_capacity(seq_len, per_token):
    """Storage-side capacity of a content-addressable KV memory (attention layers, A-002 reading).

    Attention is "memory-access decomposition" (A-002 4.5): the KV cache is a memory written on the fly,
    so its storage capacity grows with the context length, while a single readout is bounded by the
    activated content (which the flow tier of the surrounding network governs).

    Args:
        seq_len: number of stored key-value entries (context length)
        per_token: information per stored entry (nats)
    Returns:
        seq_len * per_token

    **Status: PARTIALLY VERIFIED (kv_demo, 2026-09-30, associative recall, 3 seeds, L = 2..16).**
    (i) The STORAGE tier is verified: with the per-token codebook held fixed, the memory served 8x more
    total information (L*ln C_V = 4.16 -> 33.3 nats) at accuracy 1.000 -- storage grows with the number
    of entries, independently of the per-token codebook size. (ii) The FLOW tier's mechanism is verified:
    with per_token below the payload entropy (K_tok = 4 < C_V = 8), accuracy locked at 0.48-0.52 for
    EVERY L -- adding slots did not help, so the single read is bounded by the activated payload.
    (iii) Caveat, measured: the formula is an OPTIMISTIC FLOOR for architecture design, not an achievable
    target -- the dense-memory contrast needed K = 1.5x (L=2) to 4x (L=3) more codes than L*ln C_V
    predicts, the factor growing with L, and a bottleneck placed after the query still needed >= 64 codes
    instead of ln C_V (the VQ's USABLE capacity, not the information requirement, binds in the zero-slack
    regime -- the same nominal-vs-usable gap as H(Z) tightness and the wall looseness). The pre-registered
    strict criteria were all missed (zero-slack point at L=16: 0.877 vs 0.90; constant off by 1.5-4x);
    recorded as-is in docs/kv-memory.md.
    """
    return seq_len * per_token
