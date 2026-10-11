> Chinese version: [README_cn.md](README_cn.md) — every document in this repository ships in both languages;
> the Chinese file is the English name plus `_cn`.

# Capacity Lens

**A toolkit that answers one question with a proof and one with a measurement: how much representational
capacity does this task need at minimum, and how small can a model be before the task stops fitting?**

It works by reading a trained network as a **communication channel** with a limited rate. From the reference
model's error rate it derives the task's information requirement (`R_IB`) and the smallest codebook that could
satisfy it (`K_min`) — this side is a **certified lower bound**, so it can *refute* feasibility and never
mistake itself for a promise. From a ladder of smaller students it locates the **practical wall** (`K_wall`),
the size at which accuracy actually starts to collapse — this side is a **measurement**, and the project is
explicit about everything it depends on.

The headline rule, and the only part that is a proof:

```
        log K  <  R_IB        ⇒        this configuration cannot fit the task
     (supply)   (requirement)          no training required, no counterexample possible
```

| | |
| --- | --- |
| **Version** | `capacity_lens` 0.6.1 — 10 modules, 37 exported names, depends only on PyTorch |
| **Tests** | 94 tests, 0 failures; runnable with or without pytest |
| **Evidence** | 32 experiment scripts, 21 documents in `docs/` (each with a Chinese twin), a 33-row ledger, 26 recorded datasets — all replayable **offline**, no GPU, no network |
| **Reproduce** | `python tests/run_all.py` and `python tools/reproduce_all.py` |

---

## 1. The idea

### 1.1 Why the usual numbers do not answer the question

Parameter count, nominal bit width, FLOPs and "codebook size" all describe what a model *could* carry. They do
not say what the *task* requires, so they cannot say whether a smaller model will still work. Capacity Lens
turns both sides into information quantities measured in the same unit (nats), and then compares them in the
only direction where the comparison is safe.

### 1.2 The two sides, and why only one is certified

| Side | Quantity | How it is obtained | Epistemic status |
| --- | --- | --- | --- |
| **Requirement** | `R_IB = ln C − h_b(ε) − ε·ln(C−1)` | Fano's inequality, from the reference model's test error `ε` on a `C`-class task | a **lower bound** on the task's information need, given that this reference model can do the task at all |
| **Requirement** | `K_min = max(2, ceil(e^{R_IB}))` | the smallest codebook whose `ln K` reaches `R_IB` | the codebook floor implied by the bound above |
| **Supply** | `log K` (or `d·log q`, or a topology min/sum) | read off the candidate architecture's wiring | an **upper bound** on what the candidate can pass |
| **Decision** | `log K < R_IB` | arithmetic | **certified**: supply below the requirement means the task cannot fit. Direction-safe: no experiment can overturn it |
| **Measurement** | `K_wall` = smallest `K` whose student retains ≥90% of the reference | train a ladder of students | **protocol-relative**: it depends on the training recipe, the budget, the student family, the seed |

The asymmetry is the whole point. A certified exclusion is safe to act on; a wall is a number that must be
quoted together with the conditions under which it was measured.

### 1.3 What "certified" does and does not buy

- It **does** mean: *if* the reference model genuinely does the task, then no architecture whose certified
  capacity falls short can do it — regardless of training tricks, data, or luck.
- It **does not** mean the converse. Nothing here says a configuration *will* work, or *how well*. Every
  criterion in this project is a lower bound; they refute, they never establish.
- It is **quiet until it can speak**. On a `C`-class task `K ≥ C` can never be excluded, and excluding `K = 2`
  already needs reference accuracy ≥ 0.577 (for `K = 4`, ≥ 0.807; for `K = 8`, ≥ 0.966). Below those thresholds
  the reading is *sound but empty* — which is exactly what happens with frozen or cheap witnesses.
- It is **ex-post**: `R_IB` is back-solved from a model that already works, so it calibrates a decision rather
  than predicting the size to build beforehand.
- Its **preconditions are tiered by cost**, and the tiers are worth knowing: the exclusion judgment needs
  **zero training** (the reference model's test accuracy plus the candidate's wiring arithmetic); the
  wall-geometry pre-check needs **one forward pass** of the reference; only "measure this network's capacity"
  and "locate `K*`" require inserting a bottleneck and retraining — and those two are diagnostic scaffolding,
  not part of the certified claim.

### 1.4 The witness-relative insight

The single most practically important finding in this repository:

> **The requirement reading is only as good as the model it is read from.** A weak reference lowers `R_IB`
> (a false "feasible"); witness strength materially controls how many true exclusions the rule can make.

Cheap and frozen witnesses stay *sound* — they never produce a false exclusion — but they miss exclusions
because they cannot certify much. This is why the project never treats "we ran the rule and nothing was
excluded" as good news.

### 1.5 The same standard applies to the project's own claims

Every campaign in this repository pre-registers its criteria, emits a verdict from the fixed vocabulary
(`SUPPORT` / `PARTIAL` / `MIXED` / `REFUTED` / `UNDECIDABLE` / `RESTRICTED` / `USABLE`), and keeps its
falsified expectations on the record. Results that failed are documented as prominently as results that held,
including the project's own superseded conclusions (see the ledger and the falsification archive in §9).

---

## 2. Architecture

### 2.1 Two layers, with different jobs

```
capacity_lens/            the library: pure functions + insertable layers. No datasets, no experiments,
                          no files written. Installable on its own (the distribution ships only this package).
└── imported by ──►  examples/quickstart.py        a self-contained 30-second demonstration
                     experiments/*.py              the evidence: 32 scripts that ran the campaigns
                     tests/*.py                    the library's invariants + every pre-registered criterion
                     tools/reproduce_all.py        replays all published verdicts from the recorded data
                     tools/check_wheel.py          verifies the distribution ships the library and nothing else
                     .github/workflows/tests.yml   runs all of the above (plus --selftest) on every push
```

The library never depends on the experiments; the experiments depend only on the library (plus torch and,
for two of them, torchvision). That direction is what makes the evidence checkable: each claim in §3 was
produced by a script that anyone can re-run, and most of them can be re-verified with **zero training** from
the recorded jsonl.

### 2.2 The pipeline

```
                       ┌─────────────────────────── requirement (certified) ───────────────────────────┐
   task + reference    │  reference error ε  ──Fano──►  R_IB  ──ceil(e^R_IB)──►  K_min                  │
   model (a "witness") │                                                                               │
                       └───────────────────────────────────────────────────────────────────────────────┘
                                                          │
   candidate architecture ──wiring──► capacity upper bound │        compare:  log K  <  R_IB   ⇒  refuted
   (serial min / bypass sum / MoE two-tier / KV)           │                   (no training needed)
                                                          ▼
                       ┌─────────────────────────── practical side (measured) ─────────────────────────┐
   ladder of students  │  K = 2, 4, 8, …  ──train──►  retention (acc_K / acc_ref)  ──►  K_wall (+ band)   │
   distilling the      │  never larger than the certified side's promise, always larger than K_min     │
   witness             └───────────────────────────────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
                                    preflight verdict  (five mandatory items + wall geometry;
                                    any missing item degrades the verdict instead of silently passing)
```

### 2.3 What the library exposes

| Module | What it provides |
| --- | --- |
| `bottleneck.py` | insertable bottlenecks: `VQBottleneck` (discrete codebook, `metric="euclid"` or `"cosine"`) and `GaussBottleneck` (continuous, information-regularised) |
| `capacity.py` | capacity upper bounds as pure functions: `min_capacity_serial`, `sum_capacity_parallel`, `conditional_capacity` (MoE two-tier), `kv_storage_capacity`, `entropy_of`, `mutual_info_of` |
| `training.py` | a minimal loop that handles the VQ commitment loss and optional distillation, plus `task_info_fano` (the Fano inversion at the heart of the requirement side) |
| `distill.py` | `DistillCapacity`: `min_codebook_size(R_IB)` — the distillation capacity floor |
| `quant.py` | `QuantCapacity`: `min_bits(R_IB, d)` — the per-dimension quantization floor |
| `preflight.py` | the verdict machinery: `g0_gate`, `distill_plateau_ok`, `distill_gap_k_star`, `measured_wall`, `k_wall_band`, `looseness`, `excluded_set`, `coverage`, `false_exclusions`, `ratio_verdict`, `preflight_report` |
| `geometry.py` | separability measures and `wall_geometry_probe` (discrete vs diffuse wall, zero training) |
| `monitor.py` | training diagnostics: codebook information `H(Z)`, utilization, and why no "overfitting" shortcut is offered |
| `fisher.py` | `FisherEffectiveDim` (participation ratio of the gradient Gram matrix) and `arc_length_step` |
| `__init__.py` | the 37-name public surface, and `__version__` |

### 2.4 The evidence discipline

Every experiment script in this repository follows the same shape, and the shape is what makes the claims
auditable:

- **A pre-registration in the docstring** — design, grids, criteria, verdict rule, and the iron rules
  (criteria before data; no re-rolling seeds; failures reported, not smoothed).
- **Criteria as pure functions** — the verdict logic is separated from the training and is unit-tested in
  `tests/test_experiment_criteria.py`.
- **`--selftest`** — runs those criteria on fixture data, no training.
- **Relay mode** — expensive units are checkpointed, so a run resumes instead of recomputing.
- **`--verdict-only`** — rebuilds the verdict from the recorded jsonl with **zero training**.
- **Append-only artifacts** — `<topic>.jsonl`, `<topic>_report.txt`, `<topic>/<fingerprint>/` checkpoints, so
  revisions stay visible instead of overwriting each other.
- **A ledger entry** in `docs/ledger.md` with the verdict, the numbers, and the
  recorded flaws of the criterion that produced it.

---

## 3. What it can do today, and how far each claim is established

Every row below is backed by a recorded campaign; the note named in the last column carries the design, the
numbers and the boundaries.

| # | Capability | Established state | Boundary that matters | Details |
| --- | --- | --- | --- | --- |
| 1 | **Certified exclusion** (`log K < R_IB` ⇒ cannot fit) | **Zero false exclusions** in every campaign where it was checked: 4 class counts × 3 training recipes × 2 datasets (synthetic lattices + CIFAR-10) | Necessary condition only; silent (not informative) while the witness is weak | `witness-strength.md`, `cifar10-anchor.md` |
| 2 | **The requirement reading** `R_IB`, `K_min` | Matches Monte-Carlo ground truth on families with a computable Bayes error (lattice, random-centre mixtures, 16-D mixtures) | Ex-post; a lower bound given the witness; `K ≥ C` is never excludable | `cross-task-comparability.md` |
| 3 | **The practical wall** `K_wall` (+ tolerance band) | On fine ladders the wall grows as **`C^0.603`** (R² = 0.988, n = 4 class counts) against the requirement's **`C^0.58`** — i.e. the wall and the requirement grow at essentially the same rate | It is a threshold crossing: quote it with its band (`rel_tol` 0.05/0.10/0.15); a single number can be off by 4× on a flat curve | `wall-remeasurement.md` |
| 4 | **Looseness** `ln K_wall / R_IB` | `K_wall > K_min` **always** across a 38-scenario survey, by 1.5–48× | Not a constant, not predictable from the task; it is the honest measure of how much the certified bound under-promises | `bound-looseness.md` |
| 5 | **Recipe dependence of the wall** | The wall is **protocol-relative**, and the recipe's effect is an *interaction* of task, budget and schedule: annealing collapses a deficit where the budget suffices (0.12 → −0.008) and under-trains where it does not (one wall moved the wrong way, 8 → 32); at CIFAR-10's recorded tier a fixed learning rate fails outright (retention **+0.612 vs +0.117**), **3 of 3 seeds** | There is no "best recipe" claim — only "choose the schedule together with the budget" | `budget-and-recipe.md` §6–§7, `wall-remeasurement.md` §7 |
| 6 | **Budget dependence** | **Refuted as a law**: the retention deficit is not a power law in the budget and is not monotone (0.121 → 0.069 → 0.109); it has an interior optimum | The earlier "3× the budget halves the deficit" reading is not supported as a rule (its single pairs stand) | `budget-and-recipe.md` |
| 7 | **Cross-task comparability** | Within one synthetic family the reading **orders** tasks correctly (0 material inversions); across families it does **not** — offsets reach 1.4 nats — and even within a family the invariance is **conditional on the two witnesses being of comparable quality** | Never compare readings across tasks or families; compare one reading against a codebook size | `cross-task-comparability.md` §5–§6 |
| 8 | **Witness strength controls coverage** | SUPPORT: cheap witnesses stay sound but miss exclusions; the measured cost of skipping the witness is quantified | A weak witness yields a low `R_IB` and a false "feasible" feel | `witness-strength.md` |
| 9 | **Reachable scale axes** | Sample count: MIXED (data-limited, not compute-limited). Width: SUPPORT but marginal and fragile. Resolution: SUPPORT — the reading is invariant (Δ = 0.000 saturated, 0.198 with an under-trained witness) | Only three axes are reachable offline; natural diversity and large pretrained models are not | `scale-axes.md` |
| 10 | **Distillation sizing** (`K_min`, `K*` location) | `K_min` from the witness's accuracy alone; on discrete-wall tasks the minimum feasible `K*` is locatable with an empirical margin of 1.7–3.0 | On a real task (CIFAR-10) the exclusion holds 3/3 but `K*` location does **not** transfer — a diffuse wall provides exclusion only | `cifar10-anchor.md` |
| 11 | **Quantization floor** `q_min = 2^(R_IB/d)` | Validated on a synthetic task with 3 seeds (predicted cliff 4 vs measured 3.81–3.92); QAT removes PTQ calibration artifacts | Resolving power only when `d` is small and the task's information is large; degenerates to ≈1 bit at MNIST/CIFAR scale | `mnist_demo.py`, `match-judgment-gaps.md` |
| 12 | **Wiring rules** (serial `min`, bypass `sum`) | VERIFIED on MLP; on a real Transformer the residual bypass means a VQ inside the FFN branch is **not** the supply at all (K = 2 still retains 97%; the same VQ without the bypass collapses to 0.101) | Applying the serial `min` rule literally to per-layer `log K` kills valid configurations; read the topology first | `transformer.md` |
| 13 | **Transformer exclusion pipeline** | SUPPORT: certified exclusion and a located wall run end-to-end on the pooled serial path, **and** the structural boundary is confirmed | Inside the residual stream the capacity reading does not apply | `transformer.md` |
| 14 | **Mixture-of-experts / KV conventions** | MoE two-tier `conditional_capacity` VERIFIED (route64, 6 seeds). KV storage tier VERIFIED (storage ×8 at unchanged per-token codebook); the KV **flow** tier is an optimistic floor (measured constant 1.5–4× higher) | The flow tier's anchor must be a measured dense wall, not the Fano bound | `kv-memory.md` |
| 15 | **Continuous bottlenecks** | PARTIAL (2 of 3 criteria): the two-tier reading transfers, but the strict bound over-reads usable capacity by ≈5.3× | A Gaussian channel needs the usability correction, not the raw bound | `continuous-bottleneck.md` |
| 16 | **Pretrained / frozen witnesses** | MIXED–REFUTED: a pretrained model did not replace the trained witness (the reading stays sound but loses its teeth) | Keep the witness trained and healthy (see the G0 gate) | `pretrained-witness.md` |
| 17 | **Within-family amortisation** | MIXED–REFUTED (1 of 3 criteria): a family calibration is quantitatively feasible but does not reach verdict level; exclusion is a **threshold function**, so with a small threshold margin you must measure | When the margin is small, measure; the rule predicts the *possibility* of exclusion, not its size | `family-amortisation.md` |
| 18 | **Wall geometry probe** | Zero-training classification of a task's wall (discrete vs diffuse); threshold calibrated by a dose-response experiment and prospectively validated | For an under-trained witness the representation convention is optimistic — pass the preconditions first | `output-mapping-discreteness.md` |
| 19 | **Training-time monitoring** | `H(Z)` and codebook utilization during training, in nats; deliberately **no** "overfitting" shortcut | `H(Z)` is an upper bound on `I(X;T)` and overestimates the task's information (5.36 vs 4.16) — compare capacity with the task through the Fano direction instead | `monitor.py` docstring |

### 3.1 The five-step method

Given a task and a candidate network — only step ④ is a hard judgment, the rest are its preconditions:

| Step | Content | Tool | Status |
| --- | --- | --- | --- |
| ① | Read the wiring (serial / bypass / attention / MoE) and pick the capacity formula | the wiring table in §3.2 | serial, bypass, MoE VERIFIED; attention CANDIDATE |
| ② | Measure the requirement: `R_IB` from the witness's error, then `K_min` | `task_info_fano`, `DistillCapacity` | VERIFIED, including on a real task |
| ③ | Compute the supply from the wiring | `min_capacity_serial` / `sum_capacity_parallel` / `conditional_capacity` / `kv_storage_capacity` | VERIFIED for serial, bypass, MoE |
| ④ | **Judge the exclusion** (`log K < R_IB` ⇒ cannot fit) and report `K_wall` + band for sizing | `preflight_report` | exclusion: zero false exclusions; wall: protocol-relative |
| ⑤ | When nothing is excluded, rank training configurations **within the task** | `opt_demo` | VERIFIED (regret 0 on two tasks) |

### 3.2 Wiring table (the dictionary for step ①)

| Architecture | Storage tier (does it fit?) | Flow tier (can one sample flow through?) | Basis |
| --- | --- | --- | --- |
| pure serial | `min` over bottlenecks | `min` | VERIFIED |
| bypass / residual / parallel | `sum` (each branch counts) | ≥ the largest path | VERIFIED on MLP; consequence confirmed on a Transformer |
| MoE | shared + `n_experts·log K_e` | shared + `top_k·log K_e` | VERIFIED (route64, 6 seeds) |
| attention KV | `seq_len × per-token` information | set by the readout side's flow tier | storage VERIFIED; flow tier is an optimistic floor |

---

## 4. What it cannot do

Stated plainly, because a toolkit is defined by its refusals:

- **It cannot establish feasibility.** All criteria are lower bounds; the tool refutes and measures, it never
  promises that a configuration will work.
- **It cannot predict performance**, cliff locatability, or training difficulty from any
  information-theoretic, geometric or detection-theoretic quantity — that route was tested and failed.
- **It cannot compare architectures across tasks.** Training difficulty is comparable only within one task.
- **It cannot size a model before the fact.** `R_IB` is ex-post, read from a model that already works.
- **It cannot say anything when the witness is weak or the codebook is collapsed** — the preflight gate
  refuses to produce a number instead of producing a misleading one.
- **It cannot judge "is the model overfitting?"** from `H(Z)`; that criterion was removed as unsound.
- **It has no large-scale validation.** The reachable evidence is synthetic task families + MNIST + CIFAR-10;
  natural diversity and large pretrained models are outside what this environment can measure.

---

## 5. Install, quickstart, test, reproduce

```bash
pip install -e .                          # the library (torch only)
pip install -e ".[experiments]"           # + torchvision, for the CIFAR experiment
pip wheel . --no-deps -w dist/            # a distribution: contains capacity_lens/ and the licence only
```

**Quickstart — 30 seconds, no dataset, no GPU required:**

```bash
python examples/quickstart.py
```

```
[1] reference: acc=0.977 (err=0.023)  H(Z_t)=5.17 nats
    requirement R_IB = 3.11 nats  ->  K_min = ceil(e^R_IB) = 23
[2] certified exclusion (log K < R_IB): [2, 4, 8, 16]
        student K=  32: acc=0.764  relative gap=+0.22
        student K=  64: acc=0.978  relative gap=-0.00
[3] practical wall K_wall = 64 | looseness = ln(K_wall)/R_IB = 1.34
    measured: succeeds=[64, 128, 256] fails=[2, 4, 8, 16] | coverage=1.00 | false exclusions=[]
[4] preflight: VERDICT USABLE: K*=64; practical wall K_wall=64 (certified bound sits x1.3 lower)
```

Reading it: the boundary is real because **every excluded K really failed** (`false exclusions=[]`), and the
certified side is conservative because the wall (64) sits above the floor (23) — that gap is what `looseness`
measures.

**Tests and offline reproduction:**

```bash
python tests/run_all.py                  # 94 tests, zero dependencies
python tests/run_all.py capacity         # one module
pytest tests/                            # the same tests when pytest is available
python tools/reproduce_all.py            # rebuild every published verdict from experiments/data/*.jsonl
python tools/reproduce_all.py --check    # ... and diff against the committed snapshot
python tools/check_wheel.py --selftest   # the wheel-content checker proves it can say no
python tools/check_bilingual.py          # every document has a _cn twin that matches it
```

**Continuous integration** (`.github/workflows/tests.yml`) runs the same gates on every push, with **no GPU, no
datasets and no network**: the test suite on Python 3.9 / 3.11 / 3.13; `--selftest` for every experiment script
that declares one; the offline verdict replay (a verdict that no longer rebuilds, or that changed since the
committed snapshot, fails the build); a wheel build followed by `tools/check_wheel.py` and the public
example; and `tools/check_bilingual.py`, which fails until every document has a Chinese twin that matches it.

`tools/reproduce_all.py` imports each experiment module in process (never as a subprocess — a reproduction
tool must not depend on the sandbox mode), calls its `verdict_only()`, prints the **replayed** verdict beside
the **recorded** one, classifies modules that have no offline entry point instead of assuming one, and writes
`tools/reproduce_all_snapshot.json` so two runs can be diffed. Current state: **12 replayable modules replay
successfully, 11 expose callable criteria only, 9 are the documented early generation**. No GPU, no datasets,
no network.

**Datasets** are never downloaded by the library; place them once by hand and everything stays offline-safe
(`download=True` returns immediately when the data exists):

```
experiments/data/
├── MNIST/raw/                    train-images-idx3-ubyte, train-labels-idx1-ubyte,
│                                 t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte
└── cifar-10-batches-py/          data_batch_1 … data_batch_5, test_batch, batches.meta
```

**Running the evidence scripts:**

| Mode | Budget | Emits a verdict? |
| --- | --- | --- |
| default | full data, full epochs | yes |
| `--fast` | 20–30% subset, fewer epochs | yes (same criterion logic) |
| `--quick` | minimal | no (smoke test only) |

Current-generation scripts also support `--fresh` (recompute instead of using the relay cache),
`--verdict-only` (offline re-verdict) and `--selftest` (criteria on fixture data); the three oldest suites
select a single sub-experiment with `--only`
(`mnist_demo.py`: `p1,p2,p3,p4,quant,qat,distill,distill-mnist,fisher`; `cifar_demo.py`: `c1,c2,c3,c3-cifar,c4`;
`transformer_demo.py`: `t1,t2`), e.g. `python experiments/mnist_demo.py --fast --only distill`. The nine earliest scripts
(`cifar_demo`, `cp3_demo`, `cp3_dyn`, `cp3_dyn_sens`, `cp3_multilevel`, `cph_post`, `mnist_demo`, `opt_demo`,
`transformer_demo`) predate those conventions and are **kept untouched on purpose**: their recorded artifacts
were produced by exactly this code, so rewriting them would decouple the numbers from the code that made them.
See `CONTRIBUTING.md`.

---

## 6. Using the library

**Zero-training screening** — the most common use, and the one that needs no bottleneck and no retraining:

```python
import math
from capacity_lens import task_info_fano, DistillCapacity

C = 27
acc_reference = 0.977
R_IB  = task_info_fano(1.0 - acc_reference, C)       # the task's requirement lower bound (nats)
K_min = DistillCapacity().min_codebook_size(R_IB)    # max(2, ceil(e^R_IB))

K_candidate = 16
if math.log(K_candidate) < R_IB:
    print("certainly does not fit -- no training needed")     # certified, direction-safe
```

**The full preflight verdict** (five mandatory items; any missing item degrades the verdict instead of
silently producing a number):

```python
from capacity_lens import wall_geometry_probe, preflight_report

probe = wall_geometry_probe(teacher_feats, teacher_labels)          # one forward pass, zero training
report = preflight_report(acc_t, h_z_t, k_teacher, k_list, accs, acc_min=0.95,
                          wall_probe=probe, r_ib=R_IB)
print(report["verdict"])        # "VERDICT USABLE: K*=…; practical wall K_wall=128 (certified bound sits x3.9 lower)"
print(report["wall_band"])      # {0.05: 256, 0.10: 128, 0.15: 128} — quote a wall with its band;
                                # a ≥4× swing makes the verdict say "TOLERANCE-SENSITIVE"
```

**Inserting a bottleneck** (diagnostic scaffolding — the training objective is unchanged):

```python
from capacity_lens import VQBottleneck, TrainMonitor

net.vq = VQBottleneck(d=64, n_codes=256)            # or metric="euclid", or GaussBottleneck for a channel
monitor = TrainMonitor(net.vq)
...
out, idx = net(x)
m = monitor.step(idx)                               # {"I_XT": ..., "utilization": ...} in nats
```

**Topology decides which capacity formula applies** (getting this wrong underestimates badly — measured
0.103 → 0.876 for the same serial bottlenecks plus one parallel branch):

```python
from capacity_lens import min_capacity_serial, sum_capacity_parallel, conditional_capacity, kv_storage_capacity

min_capacity_serial([log_K1, log_K2])                       # pure serial, no bypass
sum_capacity_parallel([log_K_chain, log_K_branch])          # residual / parallel
conditional_capacity(n_experts=8, per_expert=math.log(512), top_k=2, shared=math.log(256))
kv_storage_capacity(seq_len=4096, per_token=0.5)            # an optimistic floor, see kv-memory.md
```

**Distillation sizing and quantization floor:**

```python
from capacity_lens import DistillCapacity, QuantCapacity, measured_wall, k_wall_band, looseness

DistillCapacity().min_codebook_size(R_IB)       # the codebook floor
QuantCapacity().min_bits(R_ib=3.2, d=8)         # bits per dimension
k_wall, rel = measured_wall(k_list, accs, acc_ref, rel_tol=0.10)
k_wall_band(k_list, accs, acc_ref)              # the same wall at 0.05 / 0.10 / 0.15
looseness(R_IB, k_wall)                         # ln(K_wall) / R_IB
```

---

## 7. Repository map — every file and what it is for

### 7.1 `capacity_lens/` — the library (10 files, no side effects)

| File | Purpose |
| --- | --- |
| `__init__.py` | the 37-name public surface and `__version__` |
| `bottleneck.py` | `VQBottleneck` (discrete, euclid/cosine metric) and `GaussBottleneck` (continuous channel) |
| `capacity.py` | pure capacity formulas: `min_capacity_serial`, `sum_capacity_parallel`, `conditional_capacity`, `kv_storage_capacity` |
| `distill.py` | `DistillCapacity`: `min_codebook_size`, the distillation floor |
| `quant.py` | `QuantCapacity`: `min_bits(B, d)`, the quantization floor |
| `training.py` | the minimal train/evaluate loop (handles commitment loss and distillation), `entropy_of`, `mutual_info_of`, `task_info_fano`, `joint_code`, `arc_length_step` |
| `preflight.py` | the verdict machinery: `g0_gate`, `distill_plateau_ok`, `distill_gap_k_star`, `measured_wall`, `k_wall_band`, `looseness`, `excluded_set`, `coverage`, `false_exclusions`, `ratio_verdict`, `preflight_report` |
| `geometry.py` | `silhouette_by_label`, `separation_ratio`, `vq_quantization_error`, `err_1nn`, `wall_geometry_probe` |
| `monitor.py` | `TrainMonitor`: codebook information and utilization during training (and the documented absence of an overfitting shortcut) |
| `fisher.py` | `FisherEffectiveDim`, `FisherArcLength`, `arc_length_step` |

### 7.2 `experiments/` — the evidence (32 scripts, each standalone with a verdict)

**Core validation line**

| Script | Purpose |
| --- | --- |
| `mnist_demo.py` | the MNIST + MLP line: capacity tightness (P1–P3), the serial topology bound (P2), the bypass counterexample (P4), the quantization and distillation screens, the Fisher curve |
| `cifar_demo.py` | CIFAR-10 + CNN: does the lens generalize to convolutional models and harder tasks |
| `transformer_demo.py` | the small-Transformer line: tightness (T1) and the failure of `min` under a residual stream (T2) |

**The exclusion rule and its witness**

| Script | Purpose |
| --- | --- |
| `witness_demo.py` | how witness strength controls `R_IB` and the exclusion coverage (the measured cost of skipping the witness) |
| `pretrained_witness_demo.py` | whether a pretrained model can be the witness with no target-side training |
| `tf_exclusion_demo.py` | the exclusion pipeline end-to-end on a Transformer, and its structural boundary |
| `cph_demo.py` | the real-task dual-axis validation on CIFAR-10 (exclusion + whether localisation transfers) |
| `cph_post.py` | zero-training post-analysis of the CP-H axis-2 grey zone |
| `pa1_demo.py` | the P-A1-3 budget-ladder judgment (`K_t = 128` effective-capacity audit) |
| `pa3_demo.py` | the A-003 open predictions: dual-caliber discreteness and prospective wall-probe validation |
| `pa4_demo.py` | the P-A1-4 dose response: cliff sharpness versus output-mapping discreteness |

**The wall (the measured side)**

| Script | Purpose |
| --- | --- |
| `looseness_demo.py` | how much looser the certified bound is than the practical wall, across all tasks |
| `looseness_reaudit_demo.py` | the corrected pre-registration for the falsified looseness criterion, plus the ground-truth split |
| `wall_saturation_demo.py` | why the measured wall stops growing in the lattice family |
| `wall_remeasure_demo.py` | the walls re-located under the better recipe, the paired CIFAR-10 probes at two tiers, the fine-ladder growth test and its seed sweep |
| `budget_dose_demo.py` | whether the retention deficit is a power law in the training budget |
| `budget_mechanism_demo.py` | which ingredient of the recipe makes the deficit non-monotone (2×2 factorial + wall relocation) |
| `scale_c_demo.py` | the class-count axis of the scale boundary on the project's lattice family |
| `scale_c_clean_demo.py` | the same axis with the wall's ceiling held fixed (the decisive measurement) |
| `scale_axes_demo.py` | the three scale axes reachable offline: sample count, parameter width, resolution |

**Cross-task comparability**

| Script | Purpose |
| --- | --- |
| `cross_task_demo.py` | whether a common difficulty scale exists across tasks (the ledger's P-A1-2) |
| `family_b_demo.py` | a second synthetic family (random-centre mixtures) with computable ground truth |
| `family_c_demo.py` | a third family (16-dimensional mixtures) — where the invariance turned out to be conditional |
| `family_library_demo.py` | whether a within-family calibration makes the reading quasi-a-priori for a new instance |

**Other architectures and channels**

| Script | Purpose |
| --- | --- |
| `gauss_demo.py` | whether the two-tier reading and the exclusion pipeline transfer to a continuous bottleneck |
| `kv_demo.py` | the content-addressable-memory reading on associative recall |
| `moe_demo.py` | the two-tier reading on a routed expert network |
| `cp3_demo.py` | the metric intervention (architecture × VQ metric) that dissects the "architecture family" variable |
| `cp3_dyn.py`, `cp3_dyn_sens.py`, `cp3_multilevel.py` | the CP3 dynamics follow-ups: the optimizer's signal, the early-window sensitivity, the multilevel arm |
| `opt_demo.py` | the training-in-the-loop configuration optimiser (step ⑤ of the method) |

### 7.3 `tests/` — the invariants and every pre-registered criterion (10 files, 94 tests)

| File | Purpose |
| --- | --- |
| `run_all.py` | the dependency-free runner (`python tests/run_all.py [substring]`) |
| `conftest.py`, `support.py` | path bootstrap and a minimal `approx` helper, so the suite runs with or without pytest |
| `test_capacity.py` | the capacity formulas and the wiring rules |
| `test_bottleneck.py` | the bottleneck layers, including the regressions behind the Gauss campaign |
| `test_training.py` | the training glue: entropy, mutual information, Fano inversion, one end-to-end smoke test |
| `test_preflight.py` | the headline rules: the exclusion criterion, the wall, the tolerance band, the preflight degradation path |
| `test_quant_distill.py` | the quantization and distillation screens |
| `test_geometry.py` | the separability measures and the wall probe |
| `test_experiment_criteria.py` | every experiment module's pre-registered criterion functions, imported and exercised |

### 7.4 `tools/` and `examples/`

| File | Purpose |
| --- | --- |
| `tools/reproduce_all.py` | replays every published verdict from the recorded jsonl, offline, in one command |
| `tools/reproduce_all_snapshot.json` | the diffable result of the last reproduction run |
| `tools/check_wheel.py` | verifies that a built wheel ships the library and nothing else (`--selftest` proves it can say no) |
| `tools/check_bilingual.py` | verifies that every document has a `_cn` twin with the same headings, numbers, file references and code |
| `examples/quickstart.py` | the self-contained 30-second demonstration (builds its own task; imports only the package) |

### 7.5 `docs/` — 21 documents, each with a Chinese twin

**Every document ships in both languages.** The Chinese file is the English name plus `_cn`
(`docs/transformer.md` → `docs/transformer_cn.md`), and the twin must carry the same headings, the
same numbers, the same file references and the same code — `python tools/check_bilingual.py` checks exactly
that, and CI fails until it passes. `LICENSE`, `CITATION.cff`, `.gitignore`, `pyproject.toml` and the recorded
evidence (`*.jsonl`, `*_report.txt`) are deliberately exempt: their canonical names are read by tools, and their
content is not prose.

Read in this order if you are new:

| Order | Document | What it gives you |
| --- | --- | --- |
| 1 | `framework.md` | the theoretical frame: capacity, discriminability, training difficulty — and which of the three is a hard criterion |
| 5 | `ledger.md` | **the ledger**: a 33-row table — one row per campaign with its verdict, key numbers and a link to the document that carries the full record |
| 6 | `verification-plan.md` | the pre-registered success/failure criteria for every campaign, compressed to tables |
| 6 | `falsified.md` | the falsification and revision archive — what was refuted and what replaced it |

Then, per topic:

| Document | Topic |
| --- | --- |
| `witness-strength.md` | witness strength and its control over the requirement and coverage |
| `pretrained-witness.md` | pretrained witnesses |
| `bound-looseness.md` | the certified bound versus the practical wall |
| `kv-memory.md` | the KV/content-addressable-memory reading and its optimistic constant |
| `continuous-bottleneck.md` | the continuous-bottleneck line and its usability correction |
| `transformer.md` | the Transformer exclusion pipeline and its structural boundary |
| `family-amortisation.md` | within-family amortisation |
| `cross-task-comparability.md` | cross-task comparability, three families, and the condition on invariance |
| `class-count-axis.md` | the class-count axis |
| `scale-axes.md` | sample count, width, resolution |
| `wall-saturation.md` | why the wall looked saturated, and what actually limits it |
| `budget-and-recipe.md` | the budget axis, the mechanism behind it, and the wall's relocation |
| `wall-remeasurement.md` | the walls re-measured, the real-data probes, the growth exponent and its seed sweep |
| `cifar10-anchor.md` | the CP-H / PA1 verdict report (real task + counterexample audit) |
| `output-mapping-discreteness.md` | the raw A-003 analysis: output-mapping discreteness and cliff shape |
| `match-judgment-gaps.md` | how the task↔network match judgment could be completed further |
| `artifacts.md` | the run-artifact manifest: every jsonl, report and checkpoint directory, and which claim each one backs |

`docs/docs-zh/` holds a pre-translation Chinese snapshot of these documents; it is not committed.

### 7.6 Data and artifacts (what is committed and what is not)

| Path | Contents | Committed? |
| --- | --- | --- |
| `experiments/data/*.jsonl` | 26 top-level recorded result files — the raw material every verdict is rebuilt from — plus two CP-H reading files nested in `cph/` (751 KB in total) | **yes** — without these, `tools/reproduce_all.py` can replay nothing |
| `experiments/data/*_report.txt` | 26 append-only reports, including the revision trail of changed criteria (96 KB) | **yes** — they carry the recorded verdict text |
| `experiments/data/<topic>/<fingerprint>/` | ~2000 checkpoints (1.7 GB), keyed by a run fingerprint so a changed recipe cannot silently reuse another run's numbers | no — regenerated by re-running |
| `experiments/data/MNIST/`, `experiments/data/cifar-10-batches-py/` | the datasets (233 MB), placed by hand | no — place them yourself, see §5 |
| `reference/` | local working material: pre-translation snapshots, retired process documents, old run dumps (0.9 MB) | no — gitignored, kept only on this machine |
| `docs/`, `docs/`, `docs/` | local working material: historical snapshots and drafts | no |

### 7.7 Root files

| File | Purpose |
| --- | --- |
| `pyproject.toml` | packaging: setuptools, the dynamic version read from `capacity_lens.__version__`, extras `experiments` / `test` / `dev`, pytest configuration |
| `CHANGELOG.md` | library changes per version (0.5.0 → 0.6.1) |
| `CONTRIBUTING.md` | the rules a change must satisfy, the standard experiment shape, and the two-generation script policy |
| `CITATION.cff` | citation metadata (repository URL deliberately left out until there is one) |
| `LICENSE` | MIT |
| `.gitignore` | excludes datasets, checkpoints, run artifacts and the Chinese documentation snapshot |
| `README_cn.md`, `CONTRIBUTING_cn.md`, `CHANGELOG_cn.md` | the Chinese twins of the three root documents — same structure, numbers, file references and code, checked mechanically (§7.5 states the rule) |

---

## 8. Evidence, and how to check any claim above

1. **Find the number.** Every claim in §3 names the note that carries its design and results, and the note
   names its jsonl and report.
2. **Find the ledger entry.** `ledger.md` records the verdict, the numbers, and — importantly —
   the recorded flaws of the criterion that produced it.
3. **Replay it without training.** `python tools/reproduce_all.py` rebuilds every published verdict from the
   recorded data and prints it beside the recorded text; `--check` diffs against the committed snapshot.
4. **Re-run the criterion functions.** `tests/test_experiment_criteria.py` exercises the same pure functions
   the verdicts are assembled from.
5. **Re-run the experiment.** Each script's docstring is its pre-registration; the relay cache makes a partial
   re-run cheap, and `--fresh` recomputes everything.

## 9. Versioning, citation, licence

- **Versioning**: the library version lives in `capacity_lens/__init__.py` and is read dynamically by
  `pyproject.toml`; `CHANGELOG.md` records what changed and why. Nothing has been published to PyPI — treat
  the versions as development milestones.
- **Citation**: `CITATION.cff` carries the metadata; the repository URL is intentionally empty until a
  repository exists, because inventing one would be the kind of fabricated detail this project refuses.
- **Licence**: MIT.
