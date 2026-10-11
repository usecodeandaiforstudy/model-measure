# Plugin improvement analysis: making the "task↔network match" judgment more complete

> Basis: the project's own framework documents (`framework.md`, `output-mapping-discreteness.md`) and the
> 2026-09-30 campaign (the PA1 ladder and the CP-H dual axis; see `cifar10-anchor.md` in this same directory for
> the verdicts).
> This document answers: after reading these documents, **what is still missing, and how should it be filled, for the project (Capacity Lens) to have the ability to "judge whether a task and a network match"**.
> The core position comes first: the **negative half of the match judgment has already been validated on a real task**; the direction of improvement is not "make it able to predict more",
> but **complete the negative half, fill in the wiring vocabulary, and build the applicability precheck into the tool**.

---

## 0. Overall judgment

A-001 nails the network down as "a mapping from input to output, whose properties are defined only at the output boundary"; A-002 nails architecture evolution down as
"a large mapping → routing × local functions". The implications of these two positions for the match judgment are structural:

1. **Can it match at all (the negative half)** -- determined by the throughput of the output boundary (Fano/DPI), **already provable and already validated on a real task** (CP-H R1a/R1b 3/3);
2. **Does it match well (the positive half)** -- A-001 states plainly that "suitability is always relative to the task", and the output coordinate system is defined by the task ⇒ **there is no cross-task ruler, and in theory it should not be attempted**;
3. **Does it match smoothly (the cost half)** -- A-002's main battlefield (reachability / interference / exchange rate), **orderable only within a task** (opt_demo already implements it).

So the correct reading of "improve the project so that it can match tasks and network structures" is: **make 1 complete, make the within-task version of 3 smooth, and write the boundary of 2 explicitly into the product**. The five concrete lessons below are organized as "grounding → current gap → change candidate".

---

## 1. The biggest lesson from A-002: capacity judgment upgrades to two tiers, "storage tier + flow tier"

**Grounding**: the correction in A-002 §5.3 -- routing does not create information, but it changes the "capacity--compute--storage" exchange rate;
**the increase in a MoE's total parameters is a real capacity increase** (better at equal activation budget, usually not better at equal total parameters). The two-sided decomposition of §4.5:
attention decomposes **memory access**, MoE decomposes **parameter activation**.

**Current gap**: the project's capacity topology rules are only two -- pure serial takes `min`, a bypass takes `sum` (P4). This is a
single-tier rule designed for a **feed-forward bottleneck chain**, with no vocabulary for "routing × local functions". Using it to judge a MoE/Transformer
will be systematically wrong:

- A MoE's total capacity is **the sum of all experts' parameters** (the storage tier, additive), whereas the per-example throughput is **the narrowest point along the activated
  top-k expert path** (the flow tier, takes min);
- The attention layer's KV is a **content-addressable memory** (A-002 §3.2): context length × per-token information is its
  storage capacity, and the information of a single readout is its flow-tier capacity;
- The existing `min_capacity_serial` / `sum_capacity_parallel` are exactly the degenerate cases of these two tiers: pure serial = asks only the flow tier,
  pure parallel = asks only the storage tier.

**Change candidates** (medium cost, purely additive, does not break the existing API):

1. `capacity.py` gains **wiring-aware two-tier capacity bounds**:
   - `storage_capacity(wiring)`: the total-parameters / total-codebook-entries convention (sum over branches/experts) -- answers "can the mapping be stored at all";
   - `flow_capacity(wiring, activated)`: the activated-path convention (min along activated path) -- answers "can one example carry the task information through";
   - the wiring graph is expressed by a lightweight description (node = layer, edge = path, routing = conditional activation), and `min`/`sum` degenerate to special cases.
2. README gains a "modern-architecture wiring table" (dense serial / residual / attention / MoE -- which tier judges each, and what it judges).
3. The supply side of the match judgment should report both tiers: **"storage capacity sufficient but the flow tier insufficient" (a MoE activates too little) and "flow tier sufficient but storage insufficient"
   are two different kinds of mismatch, with different exclusion rationales** -- this is precisely the productization value.

---

## 2. The lesson from A-001: architecture is wiring ⇒ the prerequisite step of the match judgment is "reading the wiring diagram"

**Grounding**: A-001 section V practical stance #4 -- "architecture is wiring: designing a network = choosing which paths the input signal travels and how it is re-encoded
on the way to the output space"; #1 -- "every measurement is taken at the output boundary".

**Current gap**: the project's topology judgment has only P4's either-or between min/sum, and it requires the user to **judge the topology themselves first** (misuse severely
underestimates; the 0.103→0.876 lesson). The tool has not made "reading the wiring" a structured prerequisite step.

**Change candidates** (low cost, process-level):

Fix the match judgment as a **five-step criteria package**, every step of which either already exists or is an implementable increment:

| step | content | current state |
| --- | --- | --- |
| 1 judge the wiring | read the network structure → fix the capacity formula (two-tier convention, see §1) | only the min/sum either-or, missing modern vocabulary |
| 2 measure the task requirement | H(Y) hard upper bound + R_IB (Fano, from the reference model) → **K_min quantization** | exists (CP-H validated the quantization) |
| 3 measure the network supply | compute the storage/flow capacity by the wiring formula | missing the two-tier convention |
| 4 judge exclusion | supply < requirement ⇒ definitely does not fit (provable, direction-safe) | exists, holds 3/3 on a real task |
| 5 cost ordering | when exclusion does not hold, give only a **within-task** normalized-cost ordering | exists (opt_demo); the boundary must be stated clearly |

These five steps are the complete form of the "task↔network match judgment" -- **all of it falls within what the theory permits** (negation + within-task ordering),
without overstepping into promising the positive half.

---

## 3. The lesson from CP-H / P-A1-4: wall-geometry precheck -- proactively telling the user "which half applies"

**Grounding**: the new proposition produced by the campaign, P-A1-4 -- **cliff sharpness is a function of output-mapping discreteness, not a property of capacity**.
grid64: the output is perfectly separable (err_1nn=0) → the mapping is discrete → for K<64 it does not fit in one step → a cliff; CIFAR-10: the output is diffuse
(err_1nn=0.23) → the mapping is diffuse → the Fano exclusion still holds (the first step), but above that K it approaches level by level, with no cliff.
**Exclusion depends on (a) information content and location depends on (b) discreteness -- the two were long conflated into the single premise "the task has a wall".**

**Current gap**: the tool hands out "K* location" as a default feature, and the user only runs into "plateau not established, K* not locatable" after finishing a real task
(the CP-H experience). The tool does not tell the user **in advance** which half applies on this task.

**Change candidates** (low cost, zero training, following the cpe/cpf probe precedent):

1. Add a **wall-geometry probe**: read the reference model's features → err_1nn / silhouette / ceiling (existing off-the-shelf functions) → classify the task as
   - **discrete wall**: err_1nn≈0 → K* location applies (synthetic hard-wall-type tasks);
   - **diffuse wall**: err_1nn ≫ 0 → promise only exclusion (K_min + certified screening), and explicitly output "K* location is not promised for this task".
2. Product behavior: the precheck result is printed at the top of the verdict report -- **"for this task I can give you K_min and the exclusion, but do not expect me to locate K*"**,
   turning the CP-H lesson into the tool's default messaging rather than the user's own pitfall experience.
3. Incidentally: this turns P-A1-4 directly from a "ledger candidate prediction" into a "plugin feature"; after the δ dose response is run, the probe threshold can be calibrated.

---

## 4. The lesson from the E-001 methodology lesson: the verdict preflight checklist (gate/plateau ≈ router purity)

**Grounding**: the lesson of index entry E-001 -- "for any routing/partition-type experiment, one must first validate the intermediate quantity router purity, and only then look at the performance conclusion.
v1 did not check purity, so it measured the training recipe rather than the hypothesis." **The intermediate quantities a conclusion depends on must be self-checked first**; this is the most valuable discipline in the index system.

**Current gap**: the project's intermediate-quantity self-checks are scattered inside the experiment scripts (the G0 double gate, codebook utilization, the plateau precondition, the quantized gate),
but are not exposed as **part of the verdict output**. The user gets a K_min number without knowing which intermediate quantities behind it have been self-checked.

**Change candidates** (low cost, API-level):

1. The verdict result carries a **preflight checklist**, each item binarized:
   - has the teacher encoded the task? (acc gate)
   - has the codebook not collapsed? (H(Z_t) ≥ 0.8·ln K_t)
   - is the plateau established? (the top of the gap curve has converged)
   - is the gate inside the quantized interval? (the K_min regime, the CP-H calibration method)
   If any item fails → the verdict is automatically degraded to "RESTRICTED/UNDECIDABLE"; never silently give out a number.
2. Run-environment disclosure: the ±0.03 CUDA run-to-run variance (measured by CP-H) as a fixed report line -- productizing the reading-convention discipline.

---

## 5. The off-limits zone the documents draw for themselves (things that must never be done when improving; listed so as not to overstep)

1. **Do not predict "which architecture is better" (cross-task)** -- A-001: the output coordinate system is defined by the task; A-002 §5.3: architecture differences are axis-3 content. The tool's architecture statements are allowed only to say "whether it is feasible under this task / a within-task cost ordering".
2. **Do not use internal readings as quality metrics** -- the weak form of P-A1-1 has already been controlled for: non-empty but redundant, all of them lose to H(Z)/acc. A new metric must pass the cheapest arbitration before it is put into service.
3. **Do not promise sufficiency** -- "it fits ⇒ it will be good" never holds; the conservation law of A-001 §3.3 has only a disproof direction.
4. **MoE capacity convention**: total parameters count as capacity (the A-002 §5.3 correction); one must never look only at the activated path -- otherwise "trading storage for compute" would be read as "no capacity growth".

---

## 6. Implementation checklist (ordered by cost)

| # | change | cost | corresponding lesson |
| --- | --- | --- | --- |
| 1 | README/research-summary gains the "five-step match judgment" and the "wall but no cliff" boundary wording | zero | §2/§3 |
| 2 | wall-geometry probe (zero training, reuse the cpe skeleton + the existing geometry functions) | low | §3 |
| 3 | verdict preflight checklist + variance disclosure (carried in the verdict output) | low | §4 |
| 4 | `capacity.py` gains wiring-aware two-tier capacity bounds (storage/flow; min/sum degenerate to special cases) | medium | §1 |
| 5 | the P-A1-4 δ dose-response experiment (validate the proposition and calibrate the probe threshold first, then bring it into the tool) | medium | §3 |
| 6 | wiring table (dense/residual/attention/MoE item by item) | low | §1 |

**Recommended order**: 1 → 2/3 (in parallel) → 5 (research validation) → 4/6 (fix the A-002 vocabulary into the capacity module).
Item 5 is the only one that needs new experiments, and it dovetails naturally with the P-A1-4 registration in the analysis ledger -- **research before product, thresholds before features**;
this is exactly the discipline this project has always kept.
