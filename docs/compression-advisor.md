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

## Rank allocation (zero training)

`plan_rank_allocation(model, x_val, y_val, budget_params)` searches per-layer ranks on a **validation**
split using forward passes only, and returns the plan plus the budget-matched uniform reference and both
measured scores. Helpers: `uniform_rank_allocation` (water-filling into the budget), `rank_plan_accuracy`
(accuracy at truncated weights; the model is restored afterwards), `rank_parameter_count`, `MIN_RANK`.

- **Never pins a layer below `MIN_RANK`.** Rank starvation is irreversible (measured: −0.205 retention,
  0 of 6 seeds recovered by fine-tuning), so this guard is hard-coded, not advisory.
- **The budget counts the Linear layers only**, and the water-filling reference spends it in full — a
  single common rank is dragged down by the smallest cap and silently under-spends.
- **Validated locally on the ELIGIBLE instances** (those whose reference accuracy reached 0.4, i.e. the
  reference actually learned the task): the top-3 candidates ranked by this score contained the
  exhaustive best every time (gap +0.0000), Spearman rho median 0.90 — i.e. **5.3× fewer fine-tunes
  (16 → 3)**. On a 10.4M-parameter instance all 16 candidates were fine-tuned to verify this; had only
  the top-3 been run, the cost **would have been 7.1 minutes instead of 37.6**.
- **Search on validation data only.** Each candidate costs one SVD per layer plus one forward pass.

```python
plan = plan_rank_allocation(model, x_val, y_val, budget_params)
print(plan["ranks"], plan["score"], plan["uniform_score"])
```

## Strategy layer (heuristic, NOT certified)

The strict functions above can only answer "impossible" or "undecided". Between the certified floor and
the practical knee sits a wide band where a configuration is information-feasible yet practically useless,
and no bound decides it. This optional layer ranks candidates there. It reuses the measurements, changes
none of them, and every output carries `STRATEGY_DISCLAIMER`.

| Function | What it does | Measured evidence (small samples, one task family) |
| --- | --- | --- |
| `retention_curve(z_fit, y_fit, z_eval, y_eval, ks, n_classes)` | retention for several K with ONE probe training | identical to the per-K measurement (max abs delta = 0) at about 2.2x the speed |
| `shortlist_rank_allocations(model, x_val, y_val, budget, k=3)` | score same-budget allocations with one forward pass each, return the top k | on the one non-flat instance examined the top-3 contained the exhaustive best (gap +0.0000); flat instances are reported as flat |
| `budget_ladder(model, x_val, y_val)` | best allocation per budget level, plus the knee | knee at 0.10x of full-rank parameters on both instances examined |
| `choose_operating_point(curve, lam)` | budget level maximising accuracy - lam * parameter share | lam in [0.05, 0.10] buys 0.10x-0.15x for 0.5-0.75 accuracy points |

Flags worth reading: `allocation_freedom` (False when the model has nothing to allocate), `flat` (best
within 1% of uniform) and `REF_FLOOR` (below 0.4 reference accuracy the score is noise). None of this is a
bound, and none of it may be quoted as one.

## CLI

```bash
python -m capacity_lens.compress --acc 0.918 --classes 10 --h-y 2.3005 --i-lb 1.97 --k-codes 64
python -m capacity_lens.compress --selftest
```

The CLI prints the certified sections only (they need no model and no data). The measured sections
need a trained model and live in the campaign driver.
