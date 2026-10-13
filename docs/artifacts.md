# Run-artifact manifest (artifacts)

> Positioning: this file is the **manifest of the recorded evidence** -- every jsonl, report and checkpoint
> directory each campaign produced, plus the informal statements and process records that were moved out of the
> README during its rewrite. They are real and kept for traceability, but they are process records rather than
> product conventions: the README carries only the formal content, and this file is where a reader comes to find
> out which artifact backs which claim.
> Shipping note: the recorded `*.jsonl` and `*_report.txt` files **are committed** (they are what
> `tools/reproduce_all.py` replays offline); the checkpoints, the datasets and the archives under
> `docs/` are not.
>
> **Three notes about the shipped evidence** (the project does not edit recorded artifacts, so that numbers stay
> coupled to the run that produced them; the one exception is recorded below):
> 1. **Eight early reports contain Chinese text** (about 830 characters in total: `cp3_dyn_report.txt`,
>    `cp3_dyn_early_report.txt`, `cp3_dyn_sens_report.txt`, `cp3_multilevel_report.txt`, `pa4_report.txt`,
>    `moe_report.txt` and two others), and one jsonl row (`cp3_multilevel.jsonl`) carries 14 Chinese characters.
>    They were produced before the project's English-only rule for documents, and they are kept as recorded. What
>    makes them unusual *now* is not that they are Chinese: every document in this repository ships in both
>    languages (`<name>.md` plus `<name>_cn.md`), so these eight reports are the only Chinese text here that has
>    no English twin.
> 2. **One report header was neutralized, and it is the only edit ever made to a recorded report.**
>    `cph_post_report.txt` line 1 used to print the drive-qualified absolute path of the jsonl it read; it now
>    prints the repository-relative path (`experiments/data/cph/cph_readings_0d8045d2.jsonl`). The change touches
>    a machine-local path in a generated header, not a number, so nothing the report *measures* changed -- and it
>    is recorded here rather than done quietly.
> 3. **Four further mentions of the machine-local environment name were neutralized.** `cifar10-anchor.md` §3, its
>    `_cn` twin, `cph_demo.py` line 127 and `pa1_demo.py` line 18 used to name the virtual environment the campaigns
>    ran in. They now read "a single GPU machine" / "a GPU". No criterion, number, verdict or code path changed; the
>    recorded torch version and GPU model were kept.

---

## I. The theoretical statement that used to open the README (the campaign-style original, now neutralized into the current README's "Grounding")

> Treat a neural network as a signal-processing system: training = finding, in the **output space**, a suitable position for the input so that the distinction relations the task requires hold; the internal state is only a channel, and properties are defined only at the output boundary. From this three constraints are split out (an "and" relation, not three knobs): (1) **information capacity** -- how much information about the input the output boundary can carry through; (2) **distinguishability** -- whether the task-relevant differences are spread out in the output space; (3) **training difficulty** -- whether optimization can move the input to the right position, and at what cost.
>
> The three axes have been through an exhaustive test: **(1) passes** (the only axis that obtained a hard criterion), **(2) closes** (all effective content is absorbed by capacity; the negative verdicts at three positions each have a rationale), **(3) passes but is comparable only within a task**. Hence the tool's capability boundary is nailed down as: **it can judge "unlearnable / not up to the job" (disproof of a necessary condition), and cannot judge "learns well" (establishment of a sufficient condition)**.

(Phrasings such as "nailed down" and "exhaustive test" are changed into neutral factual statements in the formal README.)

## II. The full text of the former "2026-09-30 campaign comparison results" section (the campaign-narrative version; the formal README keeps only the neutral validation matrix)

| line | model | experiment | validation content |
| --- | --- | --- | --- |
| synthetic grid64 | CNN-euclid | **PA1 ladder (P-A1-3, 2026-09-30)** | **A-001 counterexample audit (verdict A: the counterexample does not hold)**: budget ladder 40→200ep × 3 seeds; the K_t=128 teacher meets the gate at 120ep for every seed and is 1.0000 at 200ep for every seed -- "capacity sufficient but cannot spread out" does not exist; it is axis 3's "thin margin + high cost" (~120ep vs ~40ep for healthy layers). **The archived "K_t=128 is invalid in the end" is overturned** (the plateau was a trajectory/budget artifact); incidentally, from 160→200ep H(Z_t) goes 4.63→4.32 while acc→1.0 (the later phase compresses non-task entropy) |
| real CIFAR-10 | CNN width96 | **CP-H axis 2 (2026-09-30)** | **the closure principle's first out-of-domain test (mixed result; the weak form survives)**: the S1 premise holds (err_1nn=0.228, silhouette=0.187; non-saturated regime confirmed); S2 passes for all ×4 (I_plug/silhouette/separation/d′ carry quality information in the knee window, rho(acc) 0.60–0.92, in contrast with the synthetic saturated regime's "constant"); S3 is gray zone throughout (rho(logK) 0.64–0.82) → **H_indep not established, H_proxy not confirmed**: no independent axis was found, but the strong "pure proxy" form does not hold either -- distinguishability and capacity are highly entangled. Arbitration observation: no reading beats H(Z)/acc |
| synthetic grid64 | CNN-euclid | **PA4 dose response (P-A1-4, 2026-09-30)** | **cliff sharpness = a function of output-mapping discreteness (verdict SUPPORT; the A-003 entry has been opened)**: δ∈{0,0.15,0.3,0.5} × 3 seeds; locatability 0.67→0.33→0.33→0.00 and sharpness 0.62→0.60→0.53→0.40, monotonically non-increasing; Q1 exclusion passes 12/12 (K_min shifts down adaptively with δ); err_1nn 0.000→0.390 monotonically rising -- **a controlled reproduction and mechanistic attribution of "wall but no cliff"**, and the `wall_geometry_probe` threshold 0.02 is formally calibrated. The first round's δ=0.5 "refutation" was a grid coverage defect (accounted for as-is) |
| synthetic route64 | CNN × oracle-routed MoE | **MOE two-tier capacity reading validation (2026-09-30, 6 seeds)** | **the two-tier formula of `conditional_capacity` is validated**: T2 storage tier passes (joint codebook utilization 84-94%); F1 flow tier holds with the **measured dense wall** as its reference (routed reaches 1.000/0.999 at flow=ln256; F1b 3/3: at equal total codebook, routing is no worse and is better at the high levels -- evidence for the A-002 §5.3 interference mechanism). **The root of the first round's pre-registered refutation = a wrong reference anchor** (the Fano lower bound ln64 underestimates the practical wall by 4×, the same family as CP-H's 32× margin): **the flow tier's anchor must be the measured wall, not the Fano lower bound** (accounted for as-is) |
| real MNIST / synthetic checkpoint pool | MLP / CNN | **PA3 (A-003 open predictions, 2026-09-30)** | **P-A3-1 prospective (SUPPORT)**: on MNIST (which took no part in calibration) the probe predicted diffuse×3 and no meaningful cliff was measured (1/3 < 2/3) -- the positive arm passes prospectively; **P-A3-2 dual-caliber (PARTIAL)**: rho=+0.936 passes, and the two sides of 0.02 classify consistently 53/54, the only violation being an undertrained teacher that failed G0 (representation separable, readout undertrained) -- **the representation convention is optimistic for an undertrained reference model** (deployment discipline: pass G0 before probing) |

## III. The full text of the former "training-in-the-loop configuration optimizer" section (including the reading-convention revision history; the formal README keeps only one neutral paragraph)

> Built on two axes, (1) information capacity (the G0 double gate as a hard constraint) and (3) training difficulty (normalized cost as the primary objective), decoupled from the tool itself; generalization across two tasks has passed. It is not "predicting performance" but "searching for the least-cost training configuration given a task".
>
> **Training-in-the-loop configuration optimization: three-tier reward + SH multi-fidelity pruning**. Space = metric×width×lr (18 configurations for v1) → **v2 adds the teacher-codebook K_t axis** ({128,256,512,1024,2048}, 90 configurations in total; new layers are append-only, the old indices stay bit-for-bit unchanged). L0 hard constraint = the G0 double gate, L1 primary objective = cost-to-gate, L2 cheap signal = the early-window L_func. The lever = **"exhaust once → offline exact replay of any scheme"** (confirmed by the ground-truth self-check, 25/25 bit-for-bit identical). **Reading-convention revision history (three times, each time changing a verdict)**: wall clock (P5 falsified: the same configuration and the same code differed 4× merely because of the measurement window) → raw epochs-to-gate (P5 revision) → **normalized cost = epochs-to-gate ×(conv(w)+K·d)** (unified into three-way alignment with the signal/P9). **Main conclusions after the correction**: (1) **P2 flips from "not supported" to "SUPPORT" -- pruning has value** (under the cost convention, SH regret 3.93 vs 15.82 for equal-budget random search; the old record "pruning has no value" was a product of the raw-epoch convention -- that convention looked only at epochs, not at the 3.83×/15× higher cost per epoch); (2) P3's L0 hard constraint contributes on "selection"; (3) P4 has collateral damage; (4) P6 multi-fidelity breaks even. **Signal iterations**: v1.0 acc → v1.1 epochs-to-gate preferred → **v1.2 adds width normalization** (the only regret 0 under the cost convention). **v2/v2.3 K_t axis (capacity greed)**: Q1 is falsified (the K_t=128 layer is feasible 9/18; the real structure is a codebook×metric interaction); G1/R1 are falsified (the K_t=2048 codebook gate 6.10 excluded no configuration); **but greed self-limits at K_t=1024** (within-layer optimal cost 128:7.0→256:6.1→512:4.2→**1024:3.4**→2048:5.0, bottoming out and then rising again). **v3.0 second task (shape64) generalization test**: the T1 core criterion passes (the normalized signal has regret=0 across tasks), T2 SUPPORT (the 1024 turning point reproduces across tasks), T4 not supported (L_func still only a recorded quantity), T3 PARTIAL; also one data-pipeline bug fixed (seed mistakenly passed as sigma_pix) and two harness fixes, plus a methodology lesson recorded: **when comparing "within-layer optimal cost" across layers, the seed sets of the layers must be identical**.

(The complete revision trail is in section III of falsified.md.)

## IV. Run-artifact manifest (currently retained; artifacts already deleted together with the failed experiment scripts on 2026-09-30 are no longer listed)

| path | content |
| --- | --- |
| `opt_demo.jsonl` + `opt_demo/` | grid64 official data and checkpoints (90 configurations × 3 seeds) |
| `opt_demo_shape64.jsonl` + `opt_demo_shape64/` | shape64 official data and checkpoints (216 units, v3.0 final) |
| `cp3_dyn*.jsonl` / `cp3_dyn_sens*.jsonl` / `cp3_multilevel*.jsonl` + `cp3*/`, `cp3_multilevel/` | raw records of the CP3 four cells, L_func and the multilevel doses |
| `cph_0d8045d2.jsonl` / `cph_readings_0d8045d2.jsonl` / `cph/0d8045d2/` | the CP-H dual-axis campaign (CIFAR-10, 3 seeds, 42 unit checkpoints + the axis-2 readings) |
| `pa1.jsonl` / `pa1_report.txt` / `pa1/` | the PA1 budget ladder (P-A1-3 verdict A) |
| `pa4.jsonl` / `pa4_report.txt` / `pa4/` | the PA4 dose response (P-A1-4 verdict SUPPORT, 4 levels × 3 seeds) |
| `moe.jsonl` / `moe_report.txt` / `moe/` | MOE two-tier capacity validation (route64, 6 seeds) |
| `gauss.jsonl` / `gauss_report.txt` / `gauss/` | GAUSS continuous-bottleneck exclusion pipeline and natural efficiency (grid64, 27 units + 3 probes; see continuous-bottleneck.md) |
| `pw.jsonl` / `pw_report.txt` / `pw/` | PW frozen pretrained backbone as witness (CIFAR-10, zero training on the target side; see pretrained-witness.md) |
| `family.jsonl` / `family_report.txt` / `family/` | FAMILY within-family amortized library (the CIFAR-10 class-subset family, two protocols; see family-amortisation.md) |
| `scale_c.jsonl` / `scale_c_report.txt` / `scale_c/` | SCALE-C class-count-axis scale sweep (three rounds: no-bottleneck reference / same-family large VQ / fixed ceiling; see class-count-axis.md) |
| `scale_c_clean.jsonl` / `scale_c_clean_report.txt` / `scale_c_clean/` | SCALE-C-3 fixed ceiling (`K_t=2048`) + target cache (including targets_C*.pt; see class-count-axis.md §7) |
| `wall_sat.jsonl` / `wall_sat_report.txt` / `wall_sat/` | WALL-SATURATION Monte-Carlo ground truth (step 0), the sigma arm, the KD budget probe, and the CROSS-TASK arm rows (tag `cross`); see wall-saturation.md |
| `cross_task.jsonl` / `cross_task_report.txt` | CROSS-TASK ground-truth table (scale-equivalent parameterizations) + the P-A1-2 verdict; the arm rows themselves live in `wall_sat.jsonl`; see cross-task-comparability.md |
| `scale_axes.jsonl` / `scale_axes_report.txt` / `scale_axes/` | SCALE-AXES sample count / student width / input resolution; see scale-axes.md |
| `budget_dose.jsonl` / `budget_dose_report.txt` / `budget_dose/` | BUDGET-DOSE the deficit vs the training budget (including the `arm="overfit_check"` rows); see budget-and-recipe.md |
| `budget_mechanism.jsonl` / `budget_mechanism_report.txt` / `budget_mechanism/` | BUDGET-MECHANISM the 2×2 recipe factorial (LR schedule × dead-code restart) and the wall relocation under the good recipe; see budget-and-recipe.md §6–§7 |
| `wall_remeasure.jsonl` / `wall_remeasure_report.txt` / `wall_remeasure/` | WALL-REMEASURE the project's walls re-located under cosine annealing (shape vs values, the cp3 replication, the paired CP-H CIFAR-10 recipe probe at 20 and at the recorded 120 epochs, and the fine-ladder growth test with its seed sweep); see wall-remeasurement.md |
| `family_b.jsonl` / `family_b_report.txt` / `family_b/` | FAMILY-B random-centre Gaussian mixtures (step-0 ground truth + the four training configurations); see cross-task-comparability.md §5 |
| `family_c.jsonl` / `family_c_report.txt` / `family_c/` | FAMILY-C 16-dimensional mixtures (the configuration table + three training configurations); see cross-task-comparability.md §6 |
| `witness.jsonl` / `witness_report.txt` / `witness/` | WITNESS witness ladder (clean / merged-class grid64, 5 seeds; see witness-strength.md) |
| `looseness.jsonl` / `looseness_report.txt` | LOOSENESS looseness survey (38 scenarios, zero training; see bound-looseness.md) |
| `looseness_reaudit_report.txt` | LOOSENESS-REAUDIT: D2' with the mechanical holdout + the ground-truth split (reads `looseness.jsonl`, `wall_sat.jsonl`, `cross_task.jsonl`, `scale_c_clean.jsonl`; see bound-looseness.md §6–§7) |
| (`scale_c_clean.jsonl`, rows with `arm="budget_c1000"`) | the C=1000 budget follow-up written by `scale_c_clean_demo.py --budget-probe` (see class-count-axis.md §7.4) |
| `kv.jsonl` / `kv_report.txt` / `kv/` | KV content-addressable-memory reading validation (associative recall, three arms, 93 units; see kv-memory.md) |
| `tf.jsonl` / `tf_report.txt` / `tf/` | TF-EXCLUSION exclusion pipeline and residual-bypass boundary (TinyGPT+MNIST, 45 units; see transformer.md) |

(Deleted failed-experiment artifacts: the raw records of CP1/CP2 were removed together with `cp1_demo.py`/`cp2_demo.py`; the
`detect_all*`/`cpe*`/`cpf_*`/`cpg_*` and `cp3_l_info*` of CP-D/CP-E/CP-F/CP-G were removed together with the corresponding scripts. Their conclusion archive is in section I of falsified.md.)

## V. Formalization rewrite comparison (key points)

| original phrasing (informal) | wording in the formal README |
| --- | --- |
| "the capability boundary is **nailed down** to ..." | capability boundary: it judges only the disproof of a necessary condition |
| campaign wording such as "**exhaustive test**", "**honest negative result**", "**accounted for as-is**", "**refutation**" | neutral factual statements such as "has been tested", "the result is negative", "has been recorded" |
| "**campaign**", "**judged dead**", "**overturned**", "**artifact**" | "experiment series", "the conclusion is that it does not hold", "revision" |
| the whole "2026-09-30 campaign comparison results" section | folded into neutral rows of the "experimental validation matrix" |
| the optimizer's "reading-convention revision history (three times, each time changing a verdict)" | one neutral paragraph + a pointer to this archive and falsified |
| the author's viewpoint (the full framework.md) | moved into `docs/framework.md`; the file has been deleted from the root directory |
