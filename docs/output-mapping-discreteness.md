# Analysis record A-003: the discreteness of the output mapping determines the shape of the capacity phenomenon

> **Number**: A-003 (the framework-level proposition opened by the CP-H / PA4 campaign; its ledger entry is E-019 and the prediction rows are P-A3-1 / P-A3-2 in `ledger.md`)
> **Record date**: 2026-09-30
> **Document purpose**: archive the new framework-level proposition produced by the CP-H / PA4 campaign. Section 1 records only the original intent; the assessment comes after it.
> **Annotation conventions**: 【author】 = the author's statement; 【recorder】 = the recorder's assessment or addition; 【to be tested】 = not yet settled.

---

## 1. Original analysis (the author's intent)

【author】On the real CIFAR-10 task, CP-H found: the Fano exclusion criterion transfers successfully (R1a/R1b 3/3), but the cliff (K* location) does not transfer (plateau 1/3, the only K*=128, margin 32×)——the archived conclusion "10 classes have no wall ⇒ the criterion fails" should be rewritten as "**a wall but no cliff**". From this the proposition is put forward:

> **P-A1-4: the sharpness of the cliff is a function of the discreteness of the output mapping, not a property of capacity; the Fano exclusion is independent of discreteness.**

- **Discrete output mapping** (classes are separated clusters in the output space, err_1nn ≈ 0): one step below the wall can no longer fit it → the gap curve is sharp → the cliff is localizable;
- **Diffuse output mapping** (classes are blurred overlapping regions, err_1nn ≫ 0): each extra bit of capacity buys a little accuracy → the gap curve is diffuse → no cliff, only a diffuse knee.

## 2. Assessment at a glance

| # | Proposition | Verdict | In one sentence |
|---|---|---|---|
| 1 | cliff sharpness is a function of output-mapping discreteness | **holds** (PA4 dose response) | locatability 0.67→0.33→0.33→0.00, sharpness 0.62→0.60→0.53→0.40 non-increasing with δ |
| 2 | the Fano exclusion is independent of discreteness | **holds** (PA4 Q1 12/12) | R1a/R1b all pass at the four δ doses; K_min shifts down adaptively with the reference model's error rate (35→6) |
| 3 | the wall-geometry probe can divide tasks into the discrete/diffuse domains | **holds (threshold calibrated)** | err_1nn 0.000→0.114→0.223→0.390 monotonically rising, cleanly crossing 0.02 |

## 3. The parts that hold, and their support

### 3.1 PA4 dose response (controlled reproduction, experiment number E-004)

**Design**: the grid64 image version (encoding parameters identical to the main task, unchanged) plus an "output-mapping diffuseness" knob——each sample is mixed 50/50 with a random heterogeneous-class template with probability δ, δ∈{0,0.15,0.3,0.5} × 3 seeds; the teacher is retrained per dose (R_IB/K_min adapt accordingly), and the student is distilled over the full grid ({4,6,8,32,48,64,96,128,192,256}).

| δ | locatability | median sharpness | median err_1nn | K_min range | Q1 (R1a/R1b) |
|---|---|---|---|---|---|
| 0.0 | 0.67 | 0.62 | 0.000 (discrete) | 35–64 | 3/3 pass |
| 0.15 | 0.33 | 0.60 | 0.114 (diffuse) | 28–30 | 3/3 pass |
| 0.3 | 0.33 | 0.53 | 0.223 (diffuse) | 13–14 | 3/3 pass |
| 0.5 | 0.00 | 0.40 | 0.390 (diffuse) | 6–7 | 3/3 pass |

The pre-registered criteria Q1–Q3 all pass (details in `cifar10-anchor.md` §4b and the `experiments/pa4_demo.py` docstring).

### 3.2 Corroboration on a real task (CP-H)

CIFAR-10 (err_1nn=0.228, diffuse) and grid64 (err_1nn=0, discrete) fall in the probe's two domains respectively, and their behavior agrees with the proposition: in the diffuse domain the exclusion holds (R1a/R1b 3/3) but K* is not localizable; in the discrete domain both work.

### 3.3 One protocol fix (accounted for honestly)

In PA4's first round, δ=0.5 was misjudged "refuted"——at the time K_min=6-7 was below the grid floor K=8, so there was no grid point below the bound for R1 (a coverage defect), which was misread as a violation. After extending the grid downward to {4,6} and separating "undecidable (grid defect)" from "violation" in the criterion, Q1 passed 12/12. **A defect is not a scientific conclusion.**

## 4. Places that need to be made precise

1. **The operationalization of discreteness**: the proposition asserts the discreteness of the **output space**; the current probe uses the err_1nn/silhouette of the reference model's **representation layer** as a proxy (both CP-H and PA4 do so). **P-A3-2 has been compared (partially supported)**: the two calibers have rho=+0.936, highly aligned, and 53 of 54 units are classified identically on the two sides of 0.02; the only violation is an undertrained teacher that did not pass G0 (representation separable, readout undertrained)——**the proxy is optimistic on undertrained reference models**, so deployment must first pass the G0 pre-check.
2. **The δ intervention also lowers the teacher's achievable upper limit** (acc≈0.59 at δ=0.5), and R_IB/K_min shift accordingly——what "exclusion invariance" verifies is that the criterion holds while the wall position moves, **not that the task's information content is unchanged**.
3. **The calibration granularity of the threshold 0.02**: between δ=0 and δ=0.15 there are only two readings (0.000 vs ≥0.114), and any value of 0.02 falling in between gives the correct judgement; finer calibration would require a denser δ grid (optional).
4. The operationalization of sharpness (the maximum adjacent-pair log-gap slope) is this round's pre-registered choice; other sharpness calibers (such as the logK width over which g falls from 0.5 to 0.1) have not been compared.

## 5. The precise version (testable statement)

> The sharpness of the cliff (the steepness of the gap curve / the K* locatability) varies monotonically with the **discreteness of the output mapping**; the Fano exclusion criterion (R1a/R1b) is independent of discreteness, and its wall position K_min shifts down adaptively with the reference model's error rate. The tool's product scope thus divides into two domains: **discrete-wall tasks** are promised "exclusion + K* location", **diffuse-wall tasks** are promised only "exclusion" (K_min + certified screening); before judging, run `wall_geometry_probe` first (zero training).

## 6. Testable predictions (for later comparison; P-A1-4 has already been entered into the ledger)

| # | Prediction | If it holds, it supports | How to test | Status |
|---|---|---|---|---|
| P-A1-4 | cliff sharpness is a function of output-mapping discreteness, exclusion is independent of discreteness | the core proposition of this entry | δ dose response (pa4_demo) | **compared (supported)** |
| P-A3-1 | the wall-geometry probe is **prospective**: a task judged a "diffuse wall" must fail K* location, a task judged a "discrete wall" can locate successfully | the probe's deployment value | on new tasks, probe first, then run the distillation grid, compare prediction against measurement | **compared (supported, E-005)**: MNIST (which took no part in the calibration) was predicted diffuse ×3, and the measurement found no meaningful cliff (1/3 < 2/3)——the positive arm passes prospectively; the discrete-wall task arm still awaits new data |
| P-A3-2 | the output-space caliber (logits/softmax geometry) and the representation caliber (err_1nn) give discreteness readings in the same direction | the legitimacy of the proxy | dual-caliber comparison on the same batch of models | **compared (partially supported, E-006)**: rho=+0.936 passes; classification on the two sides of 0.02 agrees 53/54——the only violation is an undertrained teacher that did not pass G0 (representation separable, readout undertrained): **the proxy is optimistic about undertrained teachers** |

## 7. Interfaces with the existing framework (three axes / output-space standpoint)

- **With A-000 (three axes)**: the **conflated premise** "the task's information content is enough to form a wall" is split into two——(a) R_IB large enough ⇒ the wall exists (CP-H confirms CIFAR-10 satisfies this); (b) the output mapping is discrete ⇒ the cliff is sharp and localizable (synthetic-specific). **The exclusion depends on (a), the K* locator implicitly depends on (b)**. The applicability-domain statement of axis ① is refined to "exclusion is universal, location requires a hard wall (discrete mapping)".
- **With A-001 (output space)**: discreteness is a geometric property of the **output boundary**——this is one more projection of "properties are defined only at the output boundary"; the representation-layer err_1nn is merely a proxy reading of that property.
- **With A-002 (architecture decomposition)**: discreteness is a task property, independent of architecture; architecture (the content of axis ③ of A-002) can change "the cost of reaching a discrete mapping", it does not change the proposition itself.

## 8. Reference anchors

- This workspace: `docs/cifar10-anchor.md` §2/§3/§4b, `docs/framework.md` §8, `docs/framework.md` §10;
- Experiments: `experiments/pa4_demo.py` (pre-registration + implementation), `experiments/data/pa4_report.txt`, `experiments/cph_demo.py` (--read);
- Ledger: `docs/ledger.md` (P-A1-4 comparison result and E-004 registration).
