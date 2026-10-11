# Pretrained-witness note (pretrained-witness)

> Record date: 2026-09-30　Experiment: `experiments/pretrained_witness_demo.py` (pre-registered criteria in the script's docstring)
> Question: **can a pretrained model be used as a witness, with the target side not trained at
> all?** This is the only form among all of the project's witness experiments that is not
> covered, and also the most likely real usage of the tool. **Zero training on the target side**: both the reference and the measured student grids are read from the CP-H cache.

---

## 1. Design and pre-registration

Controlled pretraining: **the same architecture** (CP-H's `SmallCNN(64, 256, width=96)`), changing only the **pretraining source** ——
cross-domain (MNIST digits, grayscale→3 channels 32×32) and same-domain **without labels** (CIFAR-10 four-way rotation prediction, using no target labels).
Budget: 20% subset (10000 images), 5 epoch, Adam 1e-3; probe = **closed-form ridge regression** on standardized features (λ=1e-3·trace/d),
features = `relu(fc1(...))` before the VQ (d=64). **Target-side training amount: frozen witnesses 0; the control arms follow the original protocol.**

Criteria: **P-A** zero false exclusions for every witness; **P-B** at least one frozen pretrained witness with acc ≥ 0.65 (≈ 80% of the reference 0.80); **P-C** same-domain − cross-domain ≥ 0.10; **P-D** pretrained − random initialization ≥ 0.15; **P-E** cost–coverage table.

## 2. Results (target CIFAR-10, reference teacher acc=0.799)

| Witness | Target-side cost | acc | R_IB | K_min | Exclusion set | Coverage | False exclusions | Looseness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0 raw-pixel ridge regression | 1 second | 0.303 | 0.16 | 2 | ∅ | 0.00 | 0 | 33.3 |
| P1 **randomly initialized** backbone + probe | 0 | **0.352** | 0.23 | 2 | ∅ | 0.00 | 0 | 22.9 |
| P2 MNIST pretrained (cross-domain) + probe | 5 ep | 0.304 | 0.16 | 2 | ∅ | 0.00 | 0 | 33.1 |
| P2 same features + 1-NN | 5 ep | 0.194 | 0.04 | 2 | ∅ | 0.00 | 0 | **132.6** |
| **P3 rotation pretrained (same-domain, unlabeled)** + probe | 5 ep | **0.352** | 0.23 | 2 | ∅ | 0.00 | 0 | 22.8 |
| P3 same features + 1-NN | 5 ep | 0.217 | 0.06 | 2 | ∅ | 0.00 | 0 | 89.0 |
| P5 CNN 8 epoch on the target | 8 ep | 0.571 | 0.68 | 2 | ∅ | 0.00 | 0 | 7.8 |
| P6 CP-H teacher (reference) | 120 ep | 0.803 | 1.37 | 4 | `[2,3]` | **0.40** | 0 | 3.83 |

**Verdict = MIXED/REFUTED**: P-A **PASS**; **P-B / P-C / P-D all FAIL** —— no frozen pretrained witness reached 0.65 (best 0.352); same-domain only beats cross-domain by 0.049; **pretraining adds only 0.001 over random initialization**.

**Cross-validation**: P0 recomputed 0.303 **exactly matches W2 as recorded by witness_demo** ✓ (the protocol is consistent across scripts).

## 3. Three conclusions

1. **Reliability holds unconditionally** (P-A, 8/8 rung with zero false exclusions, including a witness with acc=0.194): **the readout never lies, it just says nothing**. The weakest witness's looseness is as high as 132.6 (the reference is 3.83), and its exclusion set is empty.
2. **"Pretraining a bit on the cheap" is ineffective on this path**: 5 epoch / 20% data pretraining adds only 0.001 over **random initialization** (0.352 vs 0.352), and is not much better than raw pixels (0.303) ⇒ frozen features (whether or not they went through such cheap pretraining) yield no exclusion at all on CIFAR-10 (coverage 0).
3. **Threshold rule (the usable product added this round)**: turn "when does a readout have content" into an exact table ——

| K to be excluded | Witness needs at least acc |
| --- | --- |
| 2 | **0.577** |
| 3 | 0.721 |
| 4 | **0.807** |
| 8 | 0.966 |
| ≥ 10 | **never possible** (`R_IB ≤ ln C = 2.303` ⇒ on a 10-class task K_min ≤ 10) |

**This table exactly explains CP-H's measured exclusion set**: the teacher's 0.803 is precisely **below** K=4's threshold 0.807 → it only excludes `{2,3}` ✓ (measured: exactly `{2,3}`). That is, the WITNESS conclusion "witness strength determines coverage" now has a **closed-form accuracy threshold**: it is not "the stronger the better", it is "below the threshold you get nothing at all".

## 4. Implications for tool usage

- **To make the tool speak, the witness must have been trained on the target task**: on this task ≥0.58 to exclude K=2, ≥0.81 to exclude K=4, ≥0.97 to exclude K=8;
- frozen features (even if they went through same-domain unlabeled pretraining) **only provide "safe silence"** —— which still has value (it will not kill anything by mistake), but cannot serve as a basis for design;
- the threshold table can be **computed first and then used**: given the task's class count C and the target K, first compute `witness_threshold(K)`, then decide "how accurately I need to train", avoiding the discovery after training that the readout is empty.

## 5. Boundaries (honest)

- what this experiment judges is the **pretraining that is affordable within this project** (5 epoch / 20% data / single-task proxy); **whether a large-scale pretrained backbone (ImageNet/CLIP level) crosses the 0.58 threshold, this experiment cannot answer** —— the offline environment has no such weights, and this is also the only untested part of the "pretrained witness" path;
- single task (CIFAR-10), single architecture (SmallCNN width 96), two readouts (probe and 1-NN); 1-NN is systematically weaker than ridge regression on frozen features (0.194–0.217 vs 0.304–0.352);
- the pretraining used augmented images (RandomCrop/flip) and deterministic rotation labels (`k = i % 4`, class-balanced);
- this experiment did no seed repetition: the pretraining is deterministic (fixed seed + cache), the readings are given by the probe's closed-form solution, and the only source of variance is the fixed seed of the data subset.
