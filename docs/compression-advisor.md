# Compression Budget Advisor

> **Module**: `capacity_lens.compress` ｜ **CLI**: `python -m capacity_lens.compress --acc 0.918`

An **advisor**, not a compressor: it never trains a compressed model and never claims a performance
gain. It answers three questions about a model you already trained — *is my budget even possible*,
*where should it be spent*, *what should I expect to lose* — and it labels every answer with the
strength of the evidence behind it.

## Evidence grades

| Grade | Meaning | Examples |
| --- | --- | --- |
| **[CERT]** | provable from a theorem (Fano's inequality, capacity accounting) | the floor `R_IB`, `K_min`, the veto, the looseness bracket |
| **[MEAS]** | measured on the instance at hand, with **zero training** of what is predicted | whole-network allocation, hard-partition retention |
| **[ARCH]** | a constant measured earlier in the campaign, quoted with its task and protocol | floor-vs-knee 10–30×, retention 0.15–0.75 |
| **[LOOSE]** | an estimate with a known looseness factor | the partition ceiling, 2–3× loose |

## Certified quantities

- `task_info_floor(acc, n_classes, h_y)` — `R_IB`, the information a C-class task at that error rate
  provably needs. **Pass the empirical `H(y)`** whenever the classes are not balanced: the default
  `ln C` overstates the floor and therefore understates looseness (`balance_warning` reports the gap).
- `capacity_floors(rib)` — codebook size, capacity in nats, and per-dimension bit floors. On
  low-information tasks (MNIST carries only `ln 10` nats) the bit floor **degenerates to one level**;
  that is a property of the bound, not a bug.
- `veto(k_codes, capacity_nats, rib)` — **IMPOSSIBLE** when the compressed information path is below
  the floor. Sound, and necessary-only: measured floors sit **10–30× below the practical knee**, so
  this gate fires only on extreme over-compression.
- `tightness_interval(i_lb, rib, h_y)` — brackets the true task-relevant looseness between
  `i_lb / R_IB` (Fano) and `H(y) / R_IB` (trivial). Registered instances: **≤1.253×** on MNIST and
  **≤1.917×** on CIFAR-10.

## Measured quantities (zero training)

- `partition_retention(z_fit, y_fit, z_eval, y_eval, k, n_classes)` — how much of `I(z;y)` survives
  hard quantisation to K codes. Fit the codebook on one split and measure on a disjoint one, with
  `len(z_eval) >= 4·K·C`, or the plug-in mutual information collapses toward zero.
  Registered: **0.65** (MNIST penultimate), **0.59** (CIFAR-10 conv trunk), **0.15–0.25** (a hard
  synthetic MLP) — i.e. **0.15–0.75 and always below 1**. Retention depends on the task *and* on which
  representation is partitioned, so a report must name the representation.
- `partition_ceiling(i_code, n_classes, h_y)` — the accuracy ceiling of any decoder that sees only a
  code carrying `i_code` nats. Sound in direction, **2–3× loose**: use it as a veto, not a prediction.

## Boundaries the module refuses to hide

1. **A per-layer proxy must never pin a layer to its minimum.** Rank starvation is irreversible
   (measured: −0.205 retention, 0 of 6 seeds recovered by fine-tuning). Bit starvation, by contrast,
   is recoverable by QAT, which is why mixed precision is allowed and helps most at tight widths.
2. **Allocate only by whole-network evaluation on a validation split** — never from a per-layer curve
   sum, never on the test set.
3. **Every wall is protocol-relative** (learning rate and schedule move it by at least 8× on record):
   quote a number together with its fingerprint.
4. **The toolkit's readings support guards and diagnoses; they do not make a model better.**

## CLI

```bash
python -m capacity_lens.compress --acc 0.918 --classes 10 --h-y 2.3005 --i-lb 1.97 --k-codes 64
python -m capacity_lens.compress --selftest
```

The CLI prints the certified sections only (they need no model and no data). The measured sections
need a trained model and live in the campaign driver.
