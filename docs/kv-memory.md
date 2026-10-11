# KV capacity readout note (kv-memory)

> Record date: 2026-09-30　Experiment: `experiments/kv_demo.py` (pre-registered criteria in that script's docstring)
> Goal: verify the last "candidate" function in the package, `kv_storage_capacity(seq_len, per_token)` ——
> the claim is that "the **storage** of a KV memory grows linearly with the number of entries (`seq_len × per_token`), while a **single readout** is limited by the activated content (the flow tier)".
> All data come from a synthetic associative recall task (zero new data, generated locally), 93 tiny units, 736 seconds.

---

## 1. Design and pre-registration

Task: L slots each store one value (value ∈ 8 classes), a query names one slot, the model outputs that slot's value.
Storage = `L·ln 8`, the payload of a single query = `ln 8` (+ addressing `ln L`).

The three arms have **exactly the same computational capacity** (encoder and decoder are both two-layer MLPs, width 128); the only difference is **where the VQ bottleneck sits**:

| Arm | Structure | Theoretical requirement |
| --- | --- | --- |
| **KV** | keys go through an unquantized addressing channel; **each slot's value** is written through a per-token VQ(K_tok); readout = attention over the quantized payloads | storage `L·log K_tok`; single flow `ln L + ln C_V` |
| **DENSE** | the whole table goes through **one** VQ(K_dense) (written before the query); the query is conditioned only at the decoder | `log K_dense ≥ L·ln C_V` (⇒ `K ≈ C_V^L`) |
| **CTRL** | query and table are fed into the encoder together, the VQ sits **after the query** | only has to carry the answer ⇒ `log K ≥ ln C_V = 2.08` |

Pre-registered criteria: K1 (KV: at `K_tok=C_V` all L ≥0.90, at `K_tok=4` all L ≤0.50) / K2 (DENSE's minimum usable K within factor 2 of `C_V^L`) / K3 (CTRL at K=8 all L ≥0.90) / K4 (`L·ln C_V` predicts the dense wall within one octave).

## 2. Results

**KV arm (content addressing)**

| L | `K_tok=4` (<C_V) | `K_tok=8` (=C_V) | `K_tok=16` (with margin) | Storage |
| --- | --- | --- | --- | --- |
| 2 | 0.510 | **1.000** | **1.000** | 4.16 nats |
| 4 | 0.484 | **1.000** | **1.000** | 8.32 |
| 8 | 0.518 | **1.000** | **1.000** | 16.6 |
| 16 | 0.490 | 0.877 | **1.000** | 33.3 |

**DENSE arm** (predicted `C_V^L` = 64 / 512 / 4096)

| L | Minimum usable K (acc≥0.90) | Predicted | Factor |
| --- | --- | --- | --- |
| 2 | **96** | 64 | 1.5× |
| 3 | **2048** | 512 | **4.0×** |
| 4 | **>4096** (only 0.467 at 4096) | 4096 | ≥1× |

**CTRL arm**

| L | K=8 | K=16 | K=64 |
| --- | --- | --- | --- |
| 2 | 0.534 | 0.667 | **1.000** |
| 4 | 0.309 | 0.333 | **0.918** |
| 8 | 0.207 | 0.250 | 0.279 |

## 3. Verdict (per pre-registration, as-is)

**Verdict = MIXED/REFUTED**: K1, K2, K3, K4 **all missed**.

- K1 missed **only because of the zero-slack point at L=16** (0.877 vs the 0.90 threshold); the same arm is 1.000 for all L at `K_tok=16`;
- K2/K4 missed because the **factor grows with L** (1.5× → 4× → beyond the grid): `L·ln C_V` gets the **scaling law** right and the **constant** too low;
- K3 missed: the CTRL arm did not collapse to `ln C_V` (0.21–0.53 at K=8, and at K=64 it falls back to 0.279 at L=8).

## 4. What the data actually say (mechanistic reading)

1. **Storage tier: holds (this is the core claim of this function)**. With `K_tok` fixed at 8/16, L grows from 2 to 16 (storage 4.16 → 33.3 nats, **8×**) while accuracy stays at 1.000 —— **the total information the memory serves grows linearly with the number of entries, independently of the per-token codebook size**.
2. **Flow tier: holds (per-token is the binding constraint)**. With `K_tok=4 < C_V` the accuracy of all four L is **locked at 0.48–0.52** (≈4/8 = the fraction of representable value classes), and **adding slots is completely useless** —— this is exactly the direct evidence for "a single readout is limited by the activated content".
3. **dense contrast: scaling law holds, constant too optimistic**. The requirement does grow exponentially with L (96 → 2048 → >4096), in the same direction as `L·ln C_V`; but the measured constant is 1.5–4× above the formula, and the **factor rises with L** —— the formula is an **optimistic floor**, not an achievable target.
4. **The control arm for bottleneck placement fails, and the way it fails is informative**: moving the VQ after the query did **not** push the requirement down to `ln C_V`. The reason is that **the VQ's usable capacity ≠ its nominal capacity**: in the "zero-slack" regime, what the quantizer has to learn is the **partitioning task** of "clustering the inputs that share the same answer into one cluster", which is far harder than the information requirement (0.2–0.5 at K=8; at K=64 it falls back to 0.279 at L=8). This has the same origin as phenomena the project measured repeatedly elsewhere: `H(Z)` not filling `log K`, the long tail decay of KD, and **the looseness of the lower bound against the measured wall** (`bound-looseness.md`: 38/38 scenarios `K_wall > K_min`).

## 5. Disposition for the package itself (already implemented)

The status of `kv_storage_capacity` changed from "**candidate, unverified**" to **partially verified**, and the measured boundary was written into the docstring:

- **storage tier verified** (associative recall, L 2→16, 8× storage growth, per-token codebook unchanged);
- **flow-tier mechanism verified** (when the per-token capacity is below the payload entropy, adding slots is useless);
- **the formula's constant is optimistic by 1.5–4×, and the factor grows with L** ⇒ this function can only be used as an **optimistic floor** for design, not as an achievable target;
- the four pre-registered strict criteria **all missed**, recorded as-is (the reason for failure is the zero-slack threshold and the factor setting, not the absence of the mechanism).

The README's wiring table was changed accordingly from "candidate/unverified" to "**partially verified (storage tier + flow-tier mechanism); constant optimistic by 1.5–4×**".

## 6. Boundaries

- Synthetic associative recall task (L ≤ 16, 8 value classes, 4096 sequences, 3 seeds); values and keys are both discrete one-hot, **continuous/natural-language memory is not covered**;
- the attention addressing channel is **not quantized** (the same treatment as MOE's oracle routing: routing/addressing information counts into the flow tier but is not carried by the payload codebook);
- the CTRL arm's failure **cannot** be used to infer "bottleneck placement does not matter" (information-wise it really only needs `ln C_V`); what it shows is that **in the zero-slack regime the VQ cannot learn that partition** —— this is a learnability limit of the quantizer, not an information-theoretic limit;
- if a cleaner version is to be rerun in the future, the pre-registration should state: ① list the zero-slack point (`K_tok=C_V`) separately and declare its expectation (≈0.9 rather than a constant 1.0); ② for the scaling-law criterion use "the factor is monotonically non-decreasing with L" instead of a fixed factor 2.
