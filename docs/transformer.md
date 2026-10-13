# Transformer exclusion pipeline and structural-boundary note (transformer)

> Record date: 2026-09-30　Experiment: `experiments/tf_exclusion_demo.py` (pre-registered criteria in that script's docstring)
> Purpose: ① run the **complete exclusion pipeline** (reference → `R_IB` → `K_min` → student grid → R1a exclusion → measured wall → looseness)
> on a **real Transformer**; ② measure the Transformer-specific **structural boundary** —— does the residual bypass mean that a
> VQ inside the FFN branch can still count as "supply"? 45 units, 91 minutes, 3 seeds.

---

## 1. Design and pre-registration

Model: the off-the-shelf `TinyGPT` (MNIST cut into 7×7 patches = 16 tokens, d48/h3/l2, causal multi-head attention + pre-LN residual, about 60k parameters).
Budget: MNIST 25% subset (15000 training), teacher and student both 20 epoch, KD (CE + KL T=3).

| Arm | Structure | Only difference |
| --- | --- | --- |
| **SER** | `k_pool=K`: the bottleneck sits on the **serial path** after pooling | pipeline arm (full 8-rung grid) |
| **FFN** | `k_ffn=K`: the bottleneck sits inside the FFN branch, **bypassed by the residual identity** | boundary arm (3 rungs) |
| **FFN-NORES** | **the same VQ**, `residual=False` | control (bypass removed) |

Criteria: **X1** (on SER the certified exclusion is true + a wall exists inside the grid), **X2** (FFN keeps ≥90% at K=2, and FFN-NORES collapses to a ≥25% gap at K=2), X3 (wiring-table consequence, reported), X4 (the Transformer family's wall and looseness).

## 2. Results

**SER arm (the three seeds are almost exactly identical)**

| K | 2 | 4 | **8** | 16 | 32 | 64 | 128 | 256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acc (seed 0) | 0.185 | 0.398 | **0.749** | 0.927 | 0.934 | 0.942 | 0.950 | 0.945 |
| Relative gap | 0.80 | 0.58 | **0.21** | 0.02 | 0.02 | 0.01 | −0.00 | 0.00 |

Reference: teacher acc **0.949 / 0.948 / 0.954**, `H(Z_t)` 4.96–5.13, `R_IB` **1.99 / 1.98 / 2.02**, `K_min` **8** (identical across the three seeds).
The certified exclusion set `{2, 4}` and the measured failure set `{2, 4}` **agree rung by rung**; the measured success set is `{16, 32, 64, 128, 256}`; **`K_wall = 16`**, `looseness` **1.38–1.40**.

**The two boundary arms (the same VQ, differing only in the bypass)**

| Arm | K=2 | K=8 | K=64 |
| --- | --- | --- | --- |
| **FFN** (with bypass) | relative gap **0.03** | 0.03 | 0.01 |
| **FFN-NORES** (no bypass) | **0.89** | 0.67 / 0.49 / 0.58 | 0.09 |

## 3. Verdict: **SUPPORT**

- **X1 3/3 PASS**: the complete exclusion pipeline runs through on a real Transformer —— the certified exclusion and the measured failure are **consistent rung by rung**, and there is a wall inside the grid (K=16);
- **X2 3/3 PASS**: the FFN arm at **K=2** (16× below `K_min=8`) still retains **97%** of the reference accuracy; **the same VQ** with the bypass removed collapses to **0.101** (relative gap 0.89);
- **X4**: `looseness = 1.4` —— **the smallest among all of the project's real-data families** (CIFAR-10 is 3.5–3.9), and consistent with the LOOSENESS law (ε≈0.05 small ⇒ looseness small).

## 4. Three readings

1. **The pipeline holds on a Transformer** (the first end-to-end evidence). The exclusion set, the failure set, the wall and the looseness are all produced on an attention architecture, and the three seeds are nearly identical rung by rung.
2. **K=8 is a live demonstration of "necessary≠sufficient"**: `log 8 = 2.079 > R_IB ≈ 1.99`, i.e. **the certified criterion does not exclude it** (the necessary condition holds), yet it retains only 76% of the reference accuracy (relative gap 0.21, falling in the gray zone) —— **it sits exactly on the certification boundary and still misses the bar**. This is not a defect, it is precisely the project's central thesis: the criterion only gives "impossible", not "good enough".
3. **The Transformer's structural boundary is confirmed, and the consequence is severe**: the residual stream **completely bypasses** the VQ inside the FFN branch, so that layer's `log K` **is not supply at all**. If one applies `min_capacity_serial` literally to a "per-layer bottleneck" reading, one would judge a network that actually works perfectly (K=2, gap 3%) as "necessarily unable to fit" —— a **false kill**. The wiring table's parallel rule (P4 already verified on MLPs) must be enforced on a real Transformer.

## 5. Interface with existing conclusions

- **P4 / T2 → this experiment**: a bypass invalidates the min bound (P4, MLP), and the residual stream keeps a narrow bottleneck inside the FFN from locking the network down (T2, +0.519) —— this experiment connects them to the **exclusion criterion** and gives the quantified consequence of the "false-kill risk";
- **LOOSENESS → this experiment**: Transformer + MNIST becomes the 4th family / 2nd architecture, and `looseness = 1.4` falls on the predicted band of "small ε ⇒ small looseness";
- **KV → this experiment**: the KV experiment measures the memory reading of "storage vs flow" (synthetic associative recall), this experiment measures the **supply reading** of "per-layer bottleneck vs bypass" (real architecture) —— the two are complementary and together delimit the supply reading of attention architectures.

## 6. Boundaries and honest accounting

- **G0 gate 2/3 did not pass**: teacher acc 0.949/0.948 misses the 0.95 threshold by only 0.001–0.002 (seed 2 at 0.954 passes). **Not re-rolled** (project discipline: no noise selection to pass a gate). An observation worth recording: the three seeds' `R_IB` differ by only 0.04 nats, and the exclusion set and the wall are **exactly the same** —— i.e. on this instance the conclusions are insensitive to the difference inside/outside the gate (the gate is a conservative precaution, not a switch for the conclusion);
- MNIST is an "easy task" (ε≈0.05, looseness 1.4), so this experiment **cannot** represent Transformer behavior in the loose regime; a Transformer version on a diffuse task is still missing (parallel to the boundary in kv-memory);
- compute: 45 units / 91 minutes; 104 seconds per unit (25% subset, 20 epoch), and the two FFN arms run only 3 rungs (the grid design fixed before the run);
- the FFN arm's K was measured only up to 64; smaller/larger rungs have no incremental information for the "the bypass imposes no constraint" conclusion.
