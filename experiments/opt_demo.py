# -*- coding: utf-8 -*-
"""opt_demo: optimizer v1 -- Successive Halving multi-fidelity pruning + three-layer reward (grid64 task).

Usage:
    python experiments/opt_demo.py --calibrate          # axis B calibration (before any search data)
    python experiments/opt_demo.py --search             # SH search (v2: 54 configs)
    python experiments/opt_demo.py --report             # T_fit ranking + feasible/infeasible boundary
    python experiments/opt_demo.py --report --rho       # also run the winner rho audit
    python experiments/opt_demo.py --main --resume      # main experiment: exhaustive gold + offline replay comparison
    general: --seeds 0,1,2  --resume  --quick (smoke, writes to isolated *_quick directory)
    (--main defaults to 0-7, 8 seeds, when --seeds is not given)
    (from v2 the space = 54 configs (including the K_t axis); indices 0-17 = v1 original space, caches reused bit-for-bit)

reward structure (v1 design approved by the user on 2026-09-23; this docstring is the pre-registration record)
--------------------------------------------------------------------
(1) Search space (v1 has 18 configs; the space cap is pre-registered = one of the anti-greedy constraints):
      metric in {euclid, cosine} x width in {16, 32, 64} x lr in {3e-4, 1e-3, 3e-3}
      width baseline W = 32 (GridCNN default), pre-registered cap width <= 2W = 64 (no wider search).
      On the v1 task side K_t = 256 is fixed (grid64, C=64) and does not enter the search space;
      **v2 (docstring (11)) adds teacher codebook K_t to the space** ({128,256,512}),
      expanding the space to 54 configs, with indices 0-17 being the v1 original space bit-for-bit unchanged.

(2) Three-layer reward (no lambda scalarization -- anti-greedy design):
      L0 hard constraint (G0 double gate): acc >= 0.95 and H(Z_t) >= 0.8*ln(256); the gate is reused from
         mnist_demo.g0_gate. Those that fail the gate are infeasible and do not participate in ranking.
      L1 primary objective T_fit: the earliest wall clock (seconds) at which the G0 double gate is reached. acc + H(Z_t) are measured every epoch,
         and the clock stops at the first double pass; not passing within 40ep = infeasible (T_fit = None).
         Wall-clock pricing = capacity rent enters the objective: wider networks are costlier per epoch, so even with fewer epochs to the gate the
         wall clock may still be costlier -- this makes "trading width for early stopping" no longer free.
         Definition: t0 is set before the training loop, and the per-epoch wall clock includes that round's evaluation overhead (observation cost is counted).
      L2 cheap screening signal L_func@10: the cumulative sum of prediction flips over the 4 probe sets in the first 10 epochs
         (verified by the CP3 series: monotone in the metric/codebook axes, per-seed separable, budget-invariant).
         Role = secondary tie-break signal for SH pruning + calibration target; it does not enter T_fit.

(3) Axis B calibration (--calibrate, before any search data; criteria before data):
      Grid: width in {16, 32, 64} (metric=euclid, lr=1e-3, 3 seeds, full 40ep curves).
      Three outputs:
        a) G0 boundary on the width side: G0 pass rate and median T_fit for each width.
        b) L_func@10 width monotonicity test (honest statement: the width ordering is a hypothesis, tested by this calibration,
           and not presumed to hold): criterion = the 3-seed median L_func@10 satisfies L(16) >= L(32) >= L(64)
           (mechanism expectation: wider representations are more stable and flip less early), with at least one strict pair.
           If not monotone -> SH pruning falls back to pure acc@rung ranking (L_func is only recorded, not used as a signal),
           and this decision precedes any search data and is honestly recorded.
        c) acc@10 / acc@20 three-seed distribution per width (SH pruning reference lines; SH actually uses ranks,
           so absolute thresholds serve only as reporting reference).

(4) SH search (--search, eta=3, rung ep10/20/40, calibrate first then search):
      rung1: 18 configs x seeds trained to 10ep; pruning score = (G0 pass count, median acc@rung,
             -median L_func@10) in lexicographic descending order, keeping the top 6. L_func takes effect only when the primary
             scores are fully tied (secondary signal, and premised on calibration (3b) passing).
      rung2: 6 configs trained to 20ep (replay definition, see (6)); same ties and rules keep the top 2.
      rung3: 2 configs trained to 40ep.
      Configs that early-stop at G0 are automatically promoted to all subsequent rungs (unit cache hit / promotion copy, no retraining).
      Ranking: feasible configs in ascending order of the 3-seed median T_fit; the winner = the smallest.
      Feasibility = at least 2 of 3 seeds pass G0 within 40ep (2/3 rule; passing with 1/3 is recorded as "fragile",
      honestly recorded).

(5) Winner rho audit (--report --rho, audit only, does not enter reward -- the third anti-greedy design):
      For the winner config, 3 seeds of teachers (early-stopped or 40ep weights) -> student distillation grid
      K in CALIBRATED_K (reusing the CP3 definition: CE+KL(T=3), 40ep, GridCNN same width and metric)
      -> distill_plateau_ok / distill_gap_k_star -> K*;
      K_min = ceil(exp(Fano(1-acc_t, C=64)));  rho = K*/K_min.
      Audit criterion (recorded honestly regardless of the result): rho in [1, 8] is regarded as reward being non-pathological
      (CP3 archived CNN grid 1.7-3.0; the wide interval only serves to catch pathology and is not a precise prediction).

(6) Checkpoint/resume and replay equivalence: each (config, seed, rung) unit seeds independently and trains from scratch up to
    the rung epoch count (replay definition); when done it writes a checkpoint + JSONL flushed line by line. Deeper rung
    units replay the prefix already trained by shallower rungs, buying "resumed runs and continuous runs being per-unit equivalent"
    (the same property as CP3); on --resume, units whose shallower rung already passed G0 are promoted by direct copy.
    Seed formulas: teacher = 16000 + 100*cfg_idx + s; student (audit) =
    17000 + 100*cfg_idx + 100*s + K (non-overlapping with the existing 4000-15000 series experiments).
    After an interruption add --resume to continue; without --resume everything is recomputed and the cache overwritten.

(7) Iron rules: print uses only GBK-encodable characters; criteria before data (this docstring is the pre-registration);
    negative results are recorded honestly; smoke (--quick) writes to an isolated *_quick directory with zero sharing with official data.

(8) Main experiment (--main): optimizer effectiveness -- the object of study is the optimizer itself, not another diagnosis using the winner.
--------------------------------------------------------------------
Motivation: the v1 search (3 seeds) exposed a real crack -- SH finally chose cfg11 (T_fit 4s) while the global optimum
is cfg1 (T_fit 3s), i.e. **pruning pruned the gold away** (cfg1 survived rung1 and dropped out at rung2).
The gap between "what the optimizer picks vs the measured optimum", and whether pruning really pays off, must be tested with a larger sample.

Key lever (offline exact replay, the foundation of this design): unit_train's early stopping is deterministic (the seed is set at the
start of the function and the pre-training RNG consumption path is independent of rung depth), so for the same (config, seed),
the rung=40 unit curve is **bit-for-bit identical** to the rung=10/20 unit within min(40, stop) epochs.
Corollary: one exhaustive run (18 configs x 8 seeds, rung=40 early stop) allows **offline exact replay** of any scheme
-- SH with different seed counts/pruning keys, equal-budget random search, single fidelity, pure acc definition -- with zero extra training.
Implementation-level equivalence is guaranteed by score_at replicating online sh_score bit-for-bit (acc_at / g0_within).

Phase 1 (compute, the only compute-consuming step): 18 configs x 8 seeds x unit_train(rung=40).
  Produces the gold table = per-config median T_fit (ground-truth ranking) and median stopped_ep (noise control).
Phase 2 (offline replay, zero training): for each scheme below gives (chosen config, median T_fit, regret, nominal cost)
  gold definition = the feasible config with the smallest median T_fit; regret = chosen T_fit - gold T_fit,
  and when the chosen one is infeasible (did not pass G0), regret = infeasible.
  Scheme set:
    exhaust   exhaustive ground truth (regret always 0, cost upper bound)
    sh8       SH protocol with 8 seeds (the optimizer this scheme advocates)
    sh3       SH protocol with 3 seeds (the definition v1 actually ran, isolating the "seed count" variable)
    sh8_noG0  SH with 8 seeds but the pruning key drops the G0 count (testing the necessity of the L0 hard constraint)
    sf10      single fidelity: all 18 configs run 10ep and 1 is selected by the rung1 score (testing "whether multi-fidelity is necessary")
    rs        equal-budget random search (same nominal cost as sh8, 200 independent trials to obtain the regret distribution)
    default   GridCNN default definition (metric=cosine, width=32, lr=1e-3 = cfg13), zero search cost

Pre-registered criteria (written on 2026-09-23; **honest statement: the 3-seed results have already been seen** (the v1 search has been run),
so the following is not a blind test but an amplified test of a known crack; the 8-seed data has not yet been produced):
  P1 seed-count effect: regret(sh8) <= regret(sh3). If sh8 still picks wrong, then the pruning-stability problem
     is unrelated to seed count; honestly recorded.
  P2 pruning value: regret(sh8) < median regret(rs) (equal-budget random search). If random search
     is no worse than SH, pruning has no value -- a key negative result, honestly recorded (honest direction: the 8-config space
     is very small, so pruning's benefit is inherently limited; this is the most likely negative result foreseeable in advance).
  P3 hard-constraint necessity: regret(sh8) <= regret(sh8_noG0). If dropping the G0 count does not make the choice worse,
     it means the L0 hard constraint contributes nothing to **selection** (it may still contribute to the validity of downstream diagnostics).
  P4 pruning collateral damage: report the survival of the gold config at each rung; if it is pruned at any rung, record collateral damage.
     It is known that with 3 seeds gold=cfg1 drops out at rung2 (the cfg11/cfg10 that pruned it are higher in acc
     but slower in T_fit -- a misalignment between the pruning score and the objective). Whether this reproduces with 8 seeds is the comparison point.
  P5 noise control: if the wall-clock T_fit ranking conflicts with the stopped_ep (epoch count) ranking, judge it as wall-clock jitter,
     switch the primary ranking to stopped_ep and note it honestly (T_fit is the objective definition, but a wall-clock
     difference on the order of 3s vs 4s may be smaller than load noise).
  P6 multi-fidelity necessity: if regret(sf10) >= regret(sh8) then multi-fidelity has value; otherwise single fidelity suffices.

Honest statement (before data): the search space has only 18 configs and the T_fit differences are extremely small (3-21s), so SH's advantage
over random search is expected to be weak; the primary value of this experiment is **quantifying regret and pruning collateral damage**,
not proving SH is better. Negative results are recorded equally.

(9) v1.1 pruning signal (epochs-to-gate first) -- one diagnosis-driven iteration (pre-registered, criteria before data)
--------------------------------------------------------------------
Diagnosis (from the (8) main experiment measurements): the v1.0 pruning score (G0 count, median acc@rung) is misaligned with the objective.
At rung1 both cfg1 and cfg11 pass the gate 8/8, so the comparison is on **accuracy level** (cfg11 already 1.000 vs
cfg1 0.972) -- but cfg1 reaches the gate at ep6 and cfg11 at ep8. The pruning score cannot measure "who reaches the gate first".
Consequence: SH/single-fidelity/random search (all three choose based on acc) all pick cfg11, and only exhaustive search picks cfg1;
that is, **the bottleneck is the signal, not the budget**.

v1.1 signal (obtainable online at the same budget, since gated units already early-stop and their stopped is known):
    score = (n_g0, -median epochs-to-gate (gated units only), median acc@rung)
Fewer epochs-to-gate ranks first; non-gated ones are automatically placed after all gated ones by n_g0.

Pre-registered criteria:
  P7 (linkage, **tautology statement**): sh8_time should finally choose cfg1 (= the epoch-count-definition gold).
     When the signal is isomorphic to the criterion, hitting it is definitionally necessary, **does not constitute independent verification**, and serves only as a linkage check.
  P8 (holdout generalization, the key test of this iteration): select with seeds 0-2, evaluate with seeds 3-7 (objective =
     holdout-set median epochs-to-gate). Prediction: the time signal gives regret = 0 on the holdout set, while the acc signal
     (v1.0) does not. If the time signal also fails on the holdout set -> the signal has no generalization value; honestly recorded.
     **Honest statement**: the 8-seed data has been seen (including some per-seed epoch-count printouts), so this test is not blind;
     the holdout split (0-2 / 3-7) was written down before computation.
  P9 (definition robustness): switch to "epochs-to-gate x width FLOPs factor" as an independent cost definition (a purely architectural quantity,
     noise-free; the width rent 16:32:64 is about 1 : 3.83 : 15.0). Prediction: gold is still cfg1
     (w16's epoch advantage is small, but w32's 3.83x cost far exceeds the 6.0 vs 7.0 difference).
     If not -> the pure epoch-count definition favors narrow nets, and a normalized definition must be used; honestly recorded.
  P10 (definition consistency): if P8 and P9 both support the time signal, then v1.1 can replace v1.0 in production;
     if either does not support it, v1.1 remains only a candidate and v1.0 is retained.

(10) v1.2 pruning signal (epochs-to-gate x width normalization) -- closing v1.1's width gap
--------------------------------------------------------------------
Diagnosis (from the (9) 8-seed online run): v1.1 uses **raw** epochs-to-gate with no width normalization.
The 8-seed rung2 surviving pair is cfg1 (w16, 6.0 epochs) and cfg3 (w32, 7.0 epochs) -- under the P9 normalized definition,
cfg3 is actually 4.5x more expensive (7.0x3.83=26.8 vs 6.0x1.00=6.0), yet it squeezed into
the final rung merely because "the raw epoch counts differ by only 1". This time the winner did not change (cfg1 wins outright), so it was a near miss; but if some wide config had 1-2 fewer epochs-to-gate,
v1.1 would pick it while the true cost is 3.83x higher. Root cause: the P5 primary definition and the v1.1 signal both use raw epoch counts,
while P9 has already proven the true cost is "epoch count x width factor" -- the three have not yet been aligned.

v1.2 signal: score = (n_g0, -median(epochs-to-gate x width factor), median acc)
    width factor = width_flops_factor(w)/width_flops_factor(16) (16:32:64 about
    1 : 3.83 : 14.98); a purely architectural quantity, with no new degrees of freedom and no new budget.

Pre-registered criteria:
  P11 (linkage, tautology statement): sh8_timenorm should finally choose cfg1. The signal is isomorphic to the criterion,
     so hitting it is necessary and serves only as a linkage check.
  P12 (key, falsifiable): after normalization the rung2 survivors should **all be w16** (prediction = {cfg1, cfg0}),
     i.e. the w32 cfg3 is stopped. If cfg3 still survives rung2, then width normalization did not work;
     honestly recorded.
  P13 (holdout generalization, normalized definition): select with seeds 0-2 -> evaluate with seeds 3-7, with the evaluation definition changed to
     **normalized cost** (epoch count x width factor). Prediction: time_norm gives regret = 0 under the normalized definition.
  P14 (conclusion stability): the winner is still cfg1, consistent with the gold of v1.1/v1.0 (v1.2 only tightens
     the boundary and does not change the champion).
  P15 (production decision): if P12 and P13 both support it -> v1.2 replaces v1.1 in production (signal, P5 primary definition,
     and P9 cost definition all aligned); otherwise v1.2 remains only a candidate.
  Honest statement: the 8-seed data has been seen, so P11/P12 are not blind (P12's specific predicted values were written down before this computation);
  P13's holdout split reuses the 0-2 / 3-7 of (9) and is not re-split.

(11) v2: expanding the search space (adding the teacher codebook K_t axis) -- striking directly at the open risk recorded in the README
--------------------------------------------------------------------
Motivation: the open item recorded in the README status section is "capacity-greedy risk (**adding codebook**/widening -> cost drops) left to
the optimizer reward design to handle". The width axis was hedged in v1.2 by cost normalization; **the codebook axis has never entered the
optimizer's action space**. The K_t axis is special: adding codebook almost does not increase compute cost (the codebook term O(K·d)
is far smaller than the convolution terms), yet it raises G0's **codebook gate** (H(Z_t) >= 0.8*ln K_t, from 4.44 to 4.99 nats when K_t goes 256->512).
So the K_t axis is the ideal probe for testing "whether the L0 hard constraint can structurally resist capacity greed":
if the signal is dragged by greed, the cost signal has no power to stop it; if the gate forms a hard boundary, greed is structurally
blocked.

Design (**single-axis expansion**, keeping the "one variable" discipline -- not adding depth or other axes at the same time):
    K_t in {128, 256, 512}; K_t=256 is the v1 original space, 18 configs **kept as-is as indices 0-17**
    (seed formulas and checkpoint paths bit-for-bit unchanged, all archived units reused, no recomputation and no migration);
    K_t=128 appended as indices 18-35 and K_t=512 as 36-53; 54 configs in total.
    Pre-registered space cap: K_t <= 512.

Pre-registered criteria (written on 2026-09-23; honest statement: the K_t=256 layer data has been seen, the 128/512 layers are entirely new):
  Q1 (lower boundary of the feasible region, falsifiable): can any config in the K_t=128 layer pass G0 (acc>=0.95 and
     H(Z_t) >= 3.88)? It is known that CNN-euclid/w32/lr1e-3 passes neither 40ep nor
     80ep in cp3_multilevel (invalid at the end state). Prediction: **the layer is entirely or almost entirely infeasible**, i.e. the lower end of the codebook axis
     is a hard boundary.
  Q2 (greedy resistance, core): does the optimizer (v1.2 signal) mindlessly take the largest on the K_t axis?
     If the K_t=512 layer is largely feasible with lower normalized cost -> the signal picks 512, showing that the **cost definition does not resist
     greed** (an explicit codebook constraint must be introduced); if the 512 layer becomes less feasible / no cheaper because the codebook gate is raised,
     then the **L0 hard constraint is structurally anti-greedy as a whole**. Either direction is honestly recorded.
  Q3 (cost definition): the capacity rent for K_t is pre-registered as "the codebook term K·d counted in multiply-add ops" -- extremely small, with **no
     artificial penalty on K_t** (keeping the criterion neutral); this is honest design: if Q2 shows the signal is dragged by K_t,
     that precisely proves geometric rent alone is insufficient.
  Q4 (conclusion stability): is the champion still in the K_t=256 layer (consistent with v1)? Yes -> the expansion axis did not change the
     conclusion, and v2's value lies in boundaries and anti-greedy evidence; No -> K_t is a genuine improvement axis; honestly recorded.
  Q5 (definition consistency): within a K_t layer, the holdout-seed test of (9)/(10) (0-2 select / 3-7 evaluate) and the
     width-normalized signal are still used; if cross-K_t-layer comparison requires a new definition (e.g. costs between layers with different codebook gates
     cannot be compared directly), note it honestly and report by layer.

Honest statement: all conclusions are still on the single grid64 task; the G0 gates differ across K_t layers, so "epochs-to-gate"
cross-layer comparison **holds only under each layer's own gate**; v2 does not change any conclusion already archived by v1.

(12) v2 wrap-up: adding codebook cost-term verification + unifying the cost definition (user ruling: greed is handled as "accepted")
--------------------------------------------------------------------
Background (v2 measurements): Q2 triggered -- the top 11 of the gold table are all K_t=512, the signal took the largest codebook,
and it was judged that "the cost definition does not resist greed". At the same time two things needing repair were exposed:
  (a) **the gold primary definition and the signal definition diverge**: the primary definition ruled by P5 is the **raw epochs-to-gate**, while the v1.2
      signal uses the **width-normalized cost**; in the 18-config space the two happen to coincide (cfg1), but they diverge after expanding to 54 configs
      (in the kt512 layer multiple widths reach the gate quickly) -- the "miss/inconsistency/
      not supported" of P11/P14/P8/P9 are all artifacts of cross-definition comparison, not real conclusions.
  (b) the rent previously contained only the convolution term, and **the codebook lookup term K·d was never counted** -- i.e. this "add the cost term" step
      had not been done before, and Q2's "an explicit codebook constraint must be introduced" was in fact derived under the premise that codebook cost was not counted.

Two things in this step (verify first, then fix the definition):
  V1 (cost-term verification, falsifiable): count the VQ lookup term into the per-epoch compute --
        per-epoch compute = conv(w) + K·d (d=64, counted in multiply-add)
     After normalization, examine: the **rent penalty percentage** of K_t 256->512 at the same width.
     Pre-registered prediction: the penalty is a **single-digit percentage** (about +3~5% for w16, smaller for w64), while the measured
     epochs-to-gate benefit is **tens of percent** (for the same triple, epochs drop by about 40%+ from 256->512) --
     the two differ by an order of magnitude, so **adding the cost term cannot penalize K_t**, and gold does not change by adding the VQ term.
     If the prediction is wrong (gold changes or the penalty reaches a double-digit percentage), then scheme 3 does not hold; honestly recorded.
  V2 (unified definition): the gold primary definition is changed to **normalized cost** (including the VQ term), unifying all three -- the v1.2 signal and the P9
     cost definition; the P5 wall-clock comparison is retained (the wall clock is still untrustworthy). P9 is changed to
     a sensitivity comparison of "conv-only rent vs rent including VQ" (which is part of V1).
     After the revision, the readings of P8/P9/P11/P14 follow the new definition; the old readings are void and honestly annotated.

User ruling: greed (adding codebook -> reaching the gate faster at almost no compute) is handled by **scheme 3** --
**accepting greed as rational optimum**, on the grounds that the objective itself is incomplete (it does not count the true cost of capacity inflation),
and geometric rent has been proven unable to add an effective penalty; explicitly restricting the action space (K_t fixed at 256) is retained as an engineering
option but not implemented in this round.

Honest statement: the above predictions were written down before this step's computation, but the conv/VQ formulas and magnitudes had already been
estimated by hand at pre-registration time, so V1 is not blind; the data (v2's 54x8 units) has been seen.

(13) v2.1: K_t=1024 -- the only measured point for the greedy upper bound
--------------------------------------------------------------------
Motivation: Q2 of (11) only proves "the signal takes the largest K_t within the space", but the space cap K_t<=512 is a **pre-registered
artificial cap**, not something blocked by physics/gates. Whether greed has a **natural upper bound** has never been tested.
The natural upper bound on the K_t axis can only come from one place: **G0's codebook gate H(Z_t) >= 0.8*ln K_t** --
for K_t 256->512 the gate is 4.44->4.99 (the 512 layer passes just the same); for K_t=1024 the gate rises to **5.545**
nats (max ln1024=6.93), i.e. it demands using up enough codewords. This is the only chance to make the gate bite on greed.

Design: append K_t=1024 as the **4th layer** (indices 54-71), space of 72 configs. The existing 0-53 indices,
seed formulas and checkpoints stay **bit-for-bit** (append only, no reordering -- this is the index discipline). The rest of the protocol is unchanged.

Pre-registered criteria (written on 2026-09-23; honest statement: the 1024 layer is entirely new, no prior):
  G1 (upper-bound feasibility, core): the G0 pass situation of the K_t=1024 layer. Prediction: the layer's **number of feasible configs
     is clearly lower than the 512 layer (18/18)** -- the codebook gate 5.545 will bite some configs (especially the narrow net
     w16, whose representation dimension is limited and hard-pressed to support the entropy of 1024 codes). If the layer still passes 18/18, then the gate
     **still does not bite** at 1024, and the greedy upper bound must rely on a higher layer or an artificial cap; honestly recorded.
  G2 (whether greed continues to pay): if the layer has feasible configs, compare its intra-layer optimal normalized cost with the
     512 layer optimum (4.20). Lower -> greed continues to pay (the upper bound is further); not lower ->
     greed has saturated near 512. Either direction is honestly recorded.
  G3 (signal behavior): record whether the v1.2 signal takes K_t=1024. Takes -> greed continues; does not take ->
     the gate/cost already forms an effective barrier.
  Honest statement: all conclusions are still on the single grid64 task; cross-layer cost comparison is constrained by "each layer having a different gate"
  ((11) Q5), so G2 serves only as a side-by-side display of intra-layer optima, not a strong conclusion.

(14) v2.2: systematic H(Z_t)/ln K_t test -- why does the gate not bite? (scheme A)
--------------------------------------------------------------------
Background (v2.1 measurements): the K_t=1024 layer is 18/18 feasible and 54/54 pass the gate; the codebook gate (0.8·ln K_t =
5.55) **does not bite**; and the fastest cfg55 (3 epochs to gate) has an entropy margin of only +1.2%. On a rough post-hoc look
H(Z_t)/ln K_t ≈ 0.82 (512 layer ~5.15/6.238, 1024 layer ~5.7/6.931), only 2~3% away from the gate coefficient
0.80 -- i.e. "the gate missing biting everywhere by just a little" may be a **coefficient-choice** problem rather than a
structural one. This step upgrades that post-hoc observation into a systematic test.

Reading definition: for each unit take **the H(Z_t) of the epoch where acc first reaches >= 0.95** (exactly the
moment the codebook gate is checked); units whose acc never reaches 0.95 take the last-epoch H and are **counted separately** (a different regime, not mixed in).
Ratio = H(Z_t)/ln K_t.

Pre-registered criteria (written on 2026-09-23; honest statement: the ratios at three points cfg36/cfg54/cfg55 have already been
seen in the v2.1 analysis, so this is **not blind**; but "whether it is approximately constant across layers" and "the maximum ratio over all units" have not been computed):
  H1 (tracking hypothesis, core): the median ratio of each K_t layer is approximately constant (inter-layer spread < 0.05).
     Holds -> H(Z_t) rises in sync with the gate 0.8·ln K_t, the coefficient 0.80 is constantly below the actual ratio,
     and the gate does not bite at **any** K_t -> greed has no natural upper bound (consistent with the v2.1 observation, and gives the mechanism).
  H2 (bitable coefficient, directly actionable): report **the minimum coefficient needed for the gate to bite** = the maximum ratio over all gated
     units. That value is the quantified answer to "how far 0.8 must be raised before it starts blocking people".
  H3 (falsification case): if the inter-layer ratio spread >= 0.05, then "tracking" does not hold, the gate behavior must be
     described layer by layer, and the greedy upper bound cannot be characterized by a single coefficient -- honestly recorded.
  Honest statement: units that did not pass the acc gate are listed separately and not mixed into H1/H2; all conclusions are still limited to grid64.
Usage: python experiments/opt_demo.py --ent-ratio --seeds 0,1,2

(15) v2.3: K_t=2048 -- does the gate **exclude** configs (rather than merely adding epochs)?
--------------------------------------------------------------------
Motivation (from the (14) measurements): the real regularity obtained after H1 was falsified is -- **the H(Z_t) at the moment acc passes the gate is approximately
constant (4.19-4.91), while the gate grows linearly as 0.8·ln K_t**. The observed global upper bound of H(Z_t)
(including the H at the 1024 layer's stopping point) is about **5.83**; while 0.8·ln 2048 = **6.10** -- i.e. the 2048 gate is
**higher than any H(Z_t) value observed so far in this task**. This is the first time the gate might genuinely "exclude" configs
(the first four layers only made it train a few more epochs). This step is the decision point.

Design: append K_t=2048 as the **5th layer** (indices 72-89), space of 90 configs. The existing 0-71 indices,
seed formulas and checkpoints stay bit-for-bit (append only, no reordering). The protocol is unchanged.

Pre-registered criteria (written on 2026-09-23; honest statement: (14)'s per-layer data and "H upper bound ~5.83" have been seen,
so this is not blind; but the 2048 layer is entirely new):
  R1 (exclusion effect, core): is the number of feasible configs in the K_t=2048 layer **lower** than the 18/18 of the 512/1024 layers?
     Prediction: **clearly lower**, based on the gate 6.10 having exceeded all observed H values so far; if H's capability
     continues to rise with K_t (about +0.55 per doubling, estimated from 512->1024 gives about 6.25 at 2048), then
     it may still pass the gate -- **both outcomes are informative**, honestly recorded:
       exclusion holds -> the gate is greed's **natural upper bound** (scheme 3's boundary is more complete);
       still not excluded -> H's capability shows no sign of saturation, and the greedy upper bound can still rely only on artificial limits.
  R2 (cost direction): does the layer's unit epochs-to-gate **rebound** (the first four layers are 6.0->4.0->3.4 monotonically decreasing)?
     Rebound -> the gate's "train a few more epochs" cost begins to exceed the speedup from the larger codebook; continues to decrease -> the speedup still dominates.
  R3 (cost-term weight): at K_t=2048 the VQ term = 64*2048 = 131072, relative to conv(w16)=483328
     it already reaches +27% (that ratio is <7% for the first four layers) -- i.e. the cost term becomes significant for the first time at the largest codebook. Record its
     effect under the normalized cost definition (whether gold is still the config of that layer).
  Honest statement: all conclusions are still limited to grid64; cross-layer comparison is constrained by each layer having a different gate ((11) Q5).
Usage: python experiments/opt_demo.py --main --seeds 0,1,2 --resume

(16) v3.0: a second task (switching the encoding modality) -- generalization test
--------------------------------------------------------------------
Motivation: after the project's scope was narrowed (see the README status section "project scope narrowed"), the only direction recommended for investment is
**task generalization**. So far all conclusions (L_func separability, cost definition, greedy self-limiting at K_t=1024, the G0
feasible-region boundary, "codebook x metric interaction") are on the single task grid64. This step switches the **encoding modality**:
position encoding -> shape encoding, with everything else (64 classes / 16x16 / the same GridCNN / noise·brightness·stripe position·
split ratio / seed scheme) **bit-for-bit unchanged**, to test whether the conclusions are task-specific.

New task shape64 (generator see make_shape64_images):
    The stripe positions are constant (position is no longer information), and each stripe (3 stripes) places one of 4 **shapes**
    (solid square/vertical bar/horizontal bar/cross) -> 4^3 = 64 classes. Discrimination must happen on the convolution **channels**
    (orientation/fill-pattern response) rather than spatial position. The only variable = the encoding modality.

Space **reduction**: collect only K_t ∈ {256, 512, 1024} (anchor layer + two greedy layers, 54 configs) --
the generalization test only needs to cover "the anchor" and "both sides of the turning point", with no need to rescan the whole axis (the 128/2048 layers are left for later if needed).

Calibration clause (before any decision data): for a new task the teacher must first be confirmed to pass the acc gate (0.95) --
whether the shape encoding can be read out after 3 rounds of 2x2 pooling is an **empirical question**. If it does not pass at 40ep, one budget
step-up is allowed (80ep, lr unchanged), and that decision precedes the decision data; if it still does not pass after the step-up -> the task is unusable for G0,
honestly recorded, and this step is terminated (it is not used for generalization conclusions).

Calibration results and budget revision (2026-09-23 initial version; **voided and recalibrated after the 2026-09-25 bug**):
  [2026-09-25 data-voiding record] the initial calibration and the first main experiment (53 units) were both based on buggy data:
  make_data's positional argument passed seed into sigma_pix (see the make_data docstring), so it actually ran as a
  "sigma sweep" (s0=sigma 0 passes all gates, s1/s2=sigma 1.0/2.0 noise drowns acc and it hangs at
  0.06/0.02) and all of them used the seed=0 noise realization. Bad data/checkpoints have all been cleared and the conclusions do not count.
  **Recalibration after the fix (2026-09-25, sigma=0.15 constant, before any shape64 decision data)**:
  the probes (euclid/lr1e-3, 3 widths, seed 0) **all pass G0**: w16 ep7 acc=0.951
  H=4.51 / w32 ep13 acc=0.959 H=4.76 / w64 ep26 acc=0.960 H=4.52
  -> the shape encoding **can be read out** after 3 rounds of pooling; the task is usable and the "task unusable" termination branch is not needed.
  But **w64 uses up 26/40 = 65% of the budget** (and the grid64 same-position slow config cfg15 precedents reaching the gate
  36/40 = 90%) -- under 40ep, a slow config's "infeasibility" would still be mixed with insufficient budget, making the T3 boundary inexplicable.
  **On this basis shape64 keeps a full budget of 80ep, rung = [20, 40, 80]** (grid64 remains 40ep,
  rung = [10, 20, 40] bit-for-bit unchanged); the reason for the step-up is revised from "w16 26ep" (bad data, void) to
  "w64 26ep + the grid64 slow-config precedent", still before any shape64 decision data.
  Incidentally recorded (cross-task consistency, weak evidence): the L_func@10 width monotonicity **does not hold for either task**
  (grid64 13.0/12.3/17.8; shape64 after the fix 13.7/19.9/19.4), so both fall back per pre-registration
  to the pure acc pruning definition; the H(Z_t) level for kt=256 is also similar (4.51-4.76, gate 4.44).
  Note: epochs-to-gate **cannot be compared across tasks** (after the fix shape readout is in fact faster: cfg1 ep6 -> ep7,
  but w64 is markedly slower ep15 -> ep26 -- the cost structure of the width axis differs between the two tasks).

Pre-registered criteria (written on 2026-09-23; honest statement: all grid64 data has been seen, shape64 is entirely new):
  T1 (signal generalization, core): does the v1.2 normalized signal still give regret = 0 on the new task (is the isomorphism of signal and
     objective task-independent).
  T2 (whether greedy self-limiting reproduces): does the intra-layer optimal normalized cost with K_t **first decrease then increase** (the turning-point location
     may differ from grid64's 1024 -- a moving turning point is evidence that "generalization holds", not a counterexample).
  T3 (whether the G0 boundary reproduces): is the K_t=256 layer (the new task's anchor) feasible, and does the **codebook x metric
     interaction** reproduce (on grid64 the K_t=128 layer is "feasible all cosine, infeasible all euclid").
  T4 (L_func early-window separability): is the early-window L_func still per-seed separable across configs (the values are not required to
     match grid64, only to be separable).
  Honest statement: the shape set and stripe positions are **my design choices** (degrees of freedom), so some T2/T3 conclusions
  may be "by design"; the shape area is partially correlated with class (9/3/3/5, some shortcut exists); all conclusions
  are still limited to synthetic tasks.

Final results (2026-09-25, **72 configs x 3 seeds = 216 units**, --layers 256,512,1024,2048,
80ep; the 2048 layer was collected additionally -- the initial version had only three layers, see the "T2 supplementary test" section below):
  Judgments (auto-output by this script's T1-T4 section):
  - **T1 supported**: the v1.2 normalized signal finally chooses = gold = cfg56(euclid,w16,lr3e-3,kt1024),
    regret = 0 -- the property that signal and objective are isomorphic **holds across tasks** (core criterion passes).
    Comparison: the pure acc signal (v1.0) has regret 2.95; v1.1 raw epoch count has regret 41.25 (it takes w64,
    with the width cost unnormalized) -- width normalization is still the decisive item, isomorphic to grid64.
  - **T2 supported (flipped after additionally collecting the 2048 layer, see the section below)**: intra-layer optimal normalized cost
    256:6.1 -> 512:6.3 -> **1024:3.4** -> 2048:5.0 -> **the turning point is at K_t=1024,
    first-decrease-then-increase holds**; greedy self-limiting at 1024 **reproduces across tasks**. The mechanism is the same as grid64:
    the intra-layer optimal config's epochs-to-gate rebounds 3 -> 4 **and** the VQ rent is +11.9% (the 1024 layer w16 rent
    1.1167 -> the 2048 layer 1.24975). The intra-layer optimal costs of the 256/1024/2048 layers are **bit-for-bit identical** to grid64
    (6.1 / 3.4 / 5.0, because the optimal config is w16 in both and the median epoch counts are the same 6/3/4),
    with only the 512 layer differing (grid64 4.2 vs this task 6.3, the optimum needs 6 epochs rather than 4).
  - **T3 partial**: the K_t=256 anchor layer is feasible 18/18 (consistent with grid64, feasibility reproduces);
    the **codebook x metric interaction is undecidable** -- the collected 256/512/1024/2048 four layers all lack "perfect separation",
    while the interaction appears on grid64 at the K_t=128 layer, which was not collected in this round.
  - **T4 not supported**: per-seed Spearman(L_func@10, normalized cost) = 0.19/0.24/0.18,
    |r| < 0.3 (weak correlation, no pruning usability); consistent with "width monotonicity does not pass" in the calibration stage.
    Conclusion: on the second task L_func still serves only as a recorded quantity and does not enter the pruning signal (v1.2 uses time_norm).
  G1/G3/R1 reproduce in the same direction as grid64 after additionally collecting the 2048 layer: G1 still has the gate not biting
  (54/54 units all pass H>=6.10), R1 still not excluded (18/18), G3's signal does not take the largest K_t
  (greed is blocked by cost rather than by the gate) -- i.e. v2's conclusion "the gate does not exclude configs, it only forms a soft upper bound through cost"
  **reproduces on the second task**. R2 remains "the upper-bound layer's median epoch count 4.0 < the 1024 layer's 5.0"
  (the all-units median across layers continues to fall; the rebound happens only on the **intra-layer optimal** branch, consistent with grid64).
  harness revisions (data already collected, recomputation only, no new criteria involved):
  - **layer-limited replay pool**: Phase 2 originally used pool=None (= all 90 configs) while --layers collected only 54,
    and n_keep was computed as 90//3^(r+1) -> the η=3 pruning ratio was implicitly loosened to ~1.8. It has been changed so that
    collect_pool()/sh_nominal_cost(pool=)/holdout/rs sampling all follow the actually collected set.
    After the fix **all conclusion directions are unchanged** (gold / sh8 / sh8_timenorm bit-for-bit identical), with only the nominal
    budget going from 21600 -> 12960 and the random-search median regret from 23.9 -> 20.0.
  - **data guard**: Q1/G3 previously lacked a guard and would read "the uncollected K_t=128/2048 layers" as "infeasible /
    blocked" (0/18 is missing data, not a negative result); they have been unified with G1/R1 as "undecidable".
  - **T1-T4 implementation**: the v3.0 criteria had previously been written only into the pre-registration and not implemented in the report; they are completed this round; among them T4's
    rank-correlation form is a **post-hoc operationalization** (the pre-registration wrote only "per-seed separable"), hereby stated.
  Foundation self-check (this project's "exhaust once -> offline replay" foundation, which must be measured rather than assumed):
  - On the first --layers run Phase 1 trained only RUNGS[-1], with no online shallow-rung control -> the whole self-check was
    skipped (not passed). It has been supplemented: actually train ep20 units in a scratch directory and compare the ep80-prefix replay score
    bit-for-bit, **3/3 identical** (cfg0/cfg1/cfg4 x seed 0, ep20 rung). scope: 3 points,
    and all are the "already gated" branch; the "not gated" branch is not covered (it requires a slow w64 config, and the long sandbox run was reclaimed).
    The same code path was 25/25 on grid64, so it is accepted; a stricter check can be run locally if needed.
T2 supplementary-test pre-registration (2026-09-25, additionally running the K_t=2048 layer so the turning point can be judged; **written before any data for that layer**):
  Definition: --layers 256,512,1024,2048 (the existing 162 units are cache hits via --resume, only the new
  2048 layer 18 configs x 3 seeds = 54 units are added). The decision branches are written down in advance:
  (a) the 2048 layer has feasible configs, and its intra-layer optimal normalized cost > the 1024 layer's 3.4 -> **the turning point is at 1024,
      T2 supported**: greedy self-limiting reproduces, the turning-point location equals grid64's (this is "generalization holds").
  (b) feasible configs exist but the cost <= 3.4 -> the monotone decrease continues up to the space upper bound, **T2 still does not reproduce**
      (the turning point is farther away or outside the space; greed has no cost self-limiting in this space).
  (c) the 2048 layer has 0 feasible configs (H(Z_t) cannot reach the gate 0.8*ln2048 = 6.10) -> **hard truncation**:
      greed is capped by the G0 gate excluding the whole layer, a mechanism **different** from v2's "soft cost upper bound", honestly recorded
      as a new conclusion (at the same time R1 flips from "not excluded" to "excluded", and G1 from "not biting" to "biting").
  Discrimination discipline: (c) must be distinguished from "not collected" -- the reporting layer uses blocked (has data but 0 feasible),
  while uncollected layers are still marked "undecidable".
  Cost note: if it falls into (c), the 54 units will run the full 80ep without early stopping, the most expensive supplementary test in this project;
  to probe the boundary first you can use --seeds 0 (single seed), and the produced units will be cache hits on the subsequent 3-seed run and not wasted.
  **Supplementary-test result: it falls into (a)** -- the 2048 layer passes G0 18/18 (H(Z_t)=6.11-6.52 vs the gate
  6.10, 54/54 units), and the intra-layer optimum cfg73(euclid,w16,lr1e-3,kt2048) has a median of 4 epochs ->
  cost 5.0 > 3.4 -> the turning point is at 1024, T2 supported (not a hard truncation).
  **[Methodological lesson, more worth recording than the result itself] when probing the boundary with a single seed we once got the opposite T2 "not reproduced" reading**
  --because the intra-layer optimum is "the minimum of the 18 configs' median epoch counts", and the 1024 layer's optimum changes with the seed count
  (1 seed: cfg65 4 epochs -> 4.5; 3 seeds: cfg56 median 3 epochs -> 3.35), while the 2048 layer's seed0
  optimum is 3 epochs -> 3.75, so the 1-seed profile 6.1/6.3/4.5/3.8 has no turning point and the 3-seed profile
  6.1/6.3/3.4/5.0 does. **Lesson: when comparing "intra-layer optimal cost" across layers, the seed sets of the layers must be
  completely identical** (only comparisons within a layer are valid, as already stated in docstring (11) Q5; but "different seed counts" is the same kind of
  contamination, which was not stated at the time). The coarse granularity of median-of-3 lets this statistic flip the criterion by jumping between 3/4 epochs,
  so a single-seed boundary probe can only answer "whether that layer is feasible" and cannot answer the turning-point location.
Usage:
  python experiments/opt_demo.py --task shape64 --calibrate --seeds 0
  # primary definition (v3.0 final: four layers; all 216 collected units are cache hits, so --replay-only can be used directly)
  python experiments/opt_demo.py --task shape64 --layers 256,512,1024,2048 \
         --main --seeds 0,1,2 --resume
  python experiments/opt_demo.py --task shape64 --layers 256,512,1024,2048 \
         --main --seeds 0,1,2 --resume --replay-only   # recompute Phase 2 only, zero training
--------------------------------------------------------------------
"""
import argparse
import json
import math
import os
import random
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data")
sys.path.append(os.path.join(_HERE, ".."))
sys.path.append(_HERE)

import torch
from torch.utils.data import DataLoader

from capacity_lens import train_one_epoch, evaluate, entropy_of, task_info_fano
from mnist_demo import distill_gap_k_star, distill_plateau_ok, g0_gate
from cp3_demo import make_data as _cp3_make_data, cell_budget, CALIBRATED_K
from cifar_demo import GridCNN

# ---------------------------------------------------------------------------
# Constants and search space (pre-registered, see docstring (1))
# ---------------------------------------------------------------------------
K_T = 256                 # v1 anchor codebook (from v2 each config carries its own kt, see docstring (11))
C_TASK = 64               # number of grid64 classes (Fano capacity term)
N_PER_CLASS = 150         # consistent with the CP3 decision runs
N_PER_CLASS_QUICK = 50    # smoke data tier (no decision)
WIDTHS = [16, 32, 64]     # W/2, W, 2W (GridCNN default width=32 = W)
LRS = [3e-4, 1e-3, 3e-3]
METRICS = ["euclid", "cosine"]
K_TEACHERS = [256, 128, 512, 1024, 2048]   # codebook axis; order fixed, new layers may only be appended (to preserve indices)


def _build_configs():
    """54 configs = the 18 of v1 (kt=256, indices 0-17 bit-for-bit unchanged, reusing all archived units and seeds)
    + K_t=128 (18-35) + K_t=512 (36-53) -- single-axis expansion, the only variable = teacher codebook."""
    out = []
    for kt in K_TEACHERS:
        for m in METRICS:
            for w in WIDTHS:
                for l in LRS:
                    out.append(dict(metric=m, width=w, lr=l, kt=kt))
    return out


CONFIGS = _build_configs()
CFG_INDEX = {(c["metric"], c["width"], c["lr"], c["kt"]): i
             for i, c in enumerate(CONFIGS)}
VQ_D = 64                 # bottleneck dimension d (codebook lookup term = K·d multiply-add/sample; docstring (12))
CALIB_CFGS = [1, 4, 7]    # the three widths of (euclid, lr=1e-3): idx1=w16 idx4=w32 idx7=w64
RUNGS = [10, 20, 40]      # official fidelity ladder; replaced by [3] in quick mode
ETA = 3                   # SH keeps 1/eta at each stage
RHO_AUDIT_LO, RHO_AUDIT_HI = 1.0, 8.0   # audit interval (docstring (5), not a precise prediction)
TEACHER_SEED_BASE = 16000
STUDENT_SEED_BASE = 17000

PROBES = {                # the 4 probe sets under the same definition as cp3_multilevel
    "P1_te512": ("te", 0, 512),
    "P2_te512b": ("te", 512, 1024),
    "P3_te1024": ("te", 0, 1024),
    "P4_tr512": ("tr", 0, 512),
}
PROBE_KEYS = list(PROBES)

QUICK = False             # set in main
COLLECT_KT = None         # --layers layer limit (None = all layers; docstring (16))


# ---------------------------------------------------------------------------
# Task switchable (docstring (16)): grid64 = the existing archived task (position encoding);
# shape64 = the second task (shape encoding, generalization test). For the default grid64 the paths and archive are bit-for-bit unchanged.
# ---------------------------------------------------------------------------
TASKS = ["grid64", "shape64"]
TASK = "grid64"          # overridden by --task in main()

SHAPE_ROWS0 = (1, 7, 12)   # stripe positions exactly as in grid64 (position no longer carries information)
SHAPE_COL0 = 6             # fixed center column


def _task_tag():
    """Task suffix: the default task gets no suffix (so the paths/seeds of the existing archive stay bit-for-bit unchanged)."""
    return "" if TASK == "grid64" else f"_{TASK}"


def make_shape64_images(n_per_class=150, sigma_pix=0.15, sigma_bright=0.1,
                        seed=0):
    """The second task shape64: 3 stripes x **4 shapes** -> 64 classes (switching the encoding modality).

    The **only difference from grid64 = the encoding modality**: each stripe's square position is constant (fixed center column),
    and the class is encoded by the **shape** placed in that stripe. So position is no longer information, and discrimination must happen on the convolution channels
    (orientation/fill-pattern response) rather than spatial position. Noise/brightness/stripe position/class count/split ratio
    are bit-for-bit the same as grid64 -- the only variable = the encoding modality; this is a "task generalization" test, not a difficulty adjustment.

    Shape set (inside a 3x3 box): 0=solid square(9px)  1=vertical bar(3px)  2=horizontal bar(3px)  3=cross(5px)
    Honest statement: the shape area is **partially correlated** with class (9/3/3/5), so a "count the pixels" shortcut partly exists;
    but the sum of the three stripes' areas is not enough to uniquely decode 64 classes, and 1 vs 2 must rely on orientation, 3 vs 1/2 on structure.
    """
    g = torch.Generator().manual_seed(seed)
    n_classes = 64
    templates = torch.zeros(n_classes, 1, 16, 16)
    for v in range(n_classes):
        v1, rem = divmod(v, 16)
        v2, v3 = divmod(rem, 4)
        for c, vi in enumerate((v1, v2, v3)):
            r0 = SHAPE_ROWS0[c]
            box = templates[v, 0, r0:r0 + 3, SHAPE_COL0:SHAPE_COL0 + 3]
            if vi == 0:        # solid square
                box[:] = 1.0
            elif vi == 1:      # vertical bar (middle column)
                box[:, 1] = 1.0
            elif vi == 2:      # horizontal bar (middle row)
                box[1, :] = 1.0
            else:              # cross
                box[1, :] = 1.0
                box[:, 1] = 1.0
    y = torch.arange(n_classes).repeat_interleave(n_per_class)
    bright = 1.0 + sigma_bright * torch.randn(len(y), 1, 1, 1, generator=g)
    x = templates[y] * bright + sigma_pix * torch.randn(len(y), 1, 16, 16,
                                                        generator=g)
    perm = torch.randperm(len(y), generator=g)
    x, y = x[perm], y[perm]
    n_train = int(0.8 * len(y))
    return (x[:n_train], y[:n_train]), (x[n_train:], y[n_train:])


def make_data(n_per_class, seed):
    """Fetch data according to the current TASK (the default grid64 reuses cp3_demo's generator, bit-for-bit same source).

    [2026-09-25 bug fix] the shape64 branch was originally make_shape64_images(n_per_class, seed)
    -- the positional argument passed seed into sigma_pix (seed stayed at the default 0): it actually ran as a sigma sweep
    (s0=sigma 0 passes all gates, s1/s2=sigma 1.0/2.0 noise drowns acc and it hangs at 0.06/0.02),
    and all data used the seed=0 noise realization. Only after switching to keyword arguments is it the pre-registered task
    (sigma=0.15 constant, the seed only changes the noise realization/split, consistent with grid64's multi-seed protocol).
    """
    if TASK == "shape64":
        return make_shape64_images(n_per_class, seed=seed)
    return _cp3_make_data("cnn", n_per_class, seed)


def _paths():
    tag = ("_quick" if QUICK else "") + _task_tag()
    return (os.path.join(_DATA, f"opt_demo{tag}.jsonl"),
            os.path.join(_DATA, f"opt_demo{tag}.log"),
            os.path.join(_DATA, f"opt_demo{tag}"))


def _calib_json():
    tag = ("_quick" if QUICK else "") + _task_tag()
    return os.path.join(_DATA, f"opt_demo{tag}_calib.json")


class _Tee:
    """Write stdout/stderr to both the terminal and the log file (host capture is unreliable)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        self.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


_LOG = None


def _redirect_log(path):
    global _LOG
    if _LOG is not None:
        try:
            _LOG.close()
        except Exception:
            pass
    _LOG = open(path, "a", encoding="utf-8")   # append: one campaign spans several runs, so the log accumulates
    sys.stdout = _Tee(sys.__stdout__, _LOG)
    sys.stderr = _Tee(sys.__stderr__, _LOG)


def _append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")
        f.flush()


def teacher_seed(cfg_idx, s):
    return TEACHER_SEED_BASE + 100 * cfg_idx + s


def cfg_tag(cfg_idx):
    c = CONFIGS[cfg_idx]
    kt = f",kt{c['kt']}" if c["kt"] != K_T else ""
    return f"cfg{cfg_idx}({c['metric']},w{c['width']},lr{c['lr']:.0e}{kt})"


def _fmt_t(t):
    return "None" if t is None else f"{t:.0f}s"


# ---------------------------------------------------------------------------
# Probe sets and L_func (same definition as cp3_multilevel)
# ---------------------------------------------------------------------------

@torch.no_grad()
def probe_pred(model, x):
    model.eval()
    out, _ = model(x)
    model.train()
    return out.argmax(dim=1)


def build_probes(xtr, xte):
    out = {}
    for k, (src, a, b) in PROBES.items():
        arr = xtr if src == "tr" else xte
        out[k] = arr[a:b]
    return out


def l_func_at10(rec):
    """The sum of L_func@10 over the 4 probe sets (when the curve has fewer than 10 epochs, take the available prefix and let the caller note it honestly)."""
    lf = rec["curve"]["l_func"]
    return sum(float(lf[k][min(9, len(lf[k]) - 1)]) for k in PROBE_KEYS)


# ---------------------------------------------------------------------------
# Core unit: (config, seed, rung) trained from scratch up to the rung epoch count, stopping when both G0 gates pass (docstring (2)/(6))
# ---------------------------------------------------------------------------

def load_unit(cfg_idx, s, rung_ep, ckpt_root):
    path = os.path.join(ckpt_root, f"cfg{cfg_idx}_s{s}_ep{rung_ep}.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


def unit_train(cfg_idx, s, rung_ep, resume, jsonl, ckpt_root, mode):
    cfg = CONFIGS[cfg_idx]
    path = os.path.join(ckpt_root, f"cfg{cfg_idx}_s{s}_ep{rung_ep}.pt")
    if resume:
        ck = load_unit(cfg_idx, s, rung_ep, ckpt_root)
        if ck is not None:
            print(f"  {cfg_tag(cfg_idx)} s{s} ep{rung_ep}: [cache hit] "
                  f"acc={ck['acc']:.3f} T_fit={_fmt_t(ck['t_fit'])}", flush=True)
            return ck
        # early-stop promotion copy: a shallower rung already passed G0 -> T_fit has been achieved, deeper fidelity adds no new information
        for prev_ep in [e for e in RUNGS if e < rung_ep]:
            prev = load_unit(cfg_idx, s, prev_ep, ckpt_root)
            if prev is None:
                continue
            if prev["t_fit"] is not None:
                prev = dict(prev)
                prev["rung_ep"] = rung_ep
                prev["promoted_from"] = prev_ep
                torch.save(prev, path)
                print(f"  {cfg_tag(cfg_idx)} s{s} ep{rung_ep}: [promoted copy from "
                      f"ep{prev_ep}] T_fit={_fmt_t(prev['t_fit'])}", flush=True)
                return prev
            break   # shallower rung did not pass the gate -> must replay training

    torch.manual_seed(teacher_seed(cfg_idx, s))
    n_pc = N_PER_CLASS_QUICK if QUICK else N_PER_CLASS
    (xtr, ytr), (xte, yte) = make_data(n_pc, s)
    train_loader = DataLoader(list(zip(xtr, ytr)), batch_size=128, shuffle=True)
    test_loader = DataLoader(list(zip(xte, yte)), batch_size=256)
    model = GridCNN(d=64, n_codes=cfg["kt"], width=cfg["width"],
                    metric=cfg["metric"])
    probes = build_probes(xtr, xte)
    preds = {k: probe_pred(model, v) for k, v in probes.items()}
    l_acc = {k: 0.0 for k in probes}
    curve = dict(acc=[], h_z=[], wall=[], l_func={k: [] for k in probes})
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    t_fit = None
    stopped = rung_ep
    t0 = time.time()
    for e in range(rung_ep):
        train_one_epoch(model, train_loader, opt)
        acc_e, idx_e = evaluate(model, test_loader)
        hz_e = entropy_of(idx_e[0], cfg["kt"])
        for k, v in probes.items():
            p = probe_pred(model, v)
            l_acc[k] += float((p != preds[k]).float().mean())
            preds[k] = p
            curve["l_func"][k].append(l_acc[k])
        wall = time.time() - t0
        curve["acc"].append(acc_e)
        curve["h_z"].append(hz_e)
        curve["wall"].append(wall)
        ok, _, _ = g0_gate(acc_e, hz_e, cfg["kt"])
        if ok:
            t_fit = wall
            stopped = e + 1
            break

    acc, h_z = curve["acc"][-1], curve["h_z"][-1]
    rec = dict(state_dict=model.state_dict(), curve=curve, t_fit=t_fit,
               stopped=stopped, rung_ep=rung_ep, acc=acc, h_z=h_z,
               cfg_idx=cfg_idx, seed=s)
    torch.save(rec, path)
    _append_jsonl(jsonl, dict(
        mode=mode, cfg_idx=cfg_idx, metric=cfg["metric"], width=cfg["width"],
        kt=cfg["kt"], lr=cfg["lr"], seed=s, rung_ep=rung_ep,
        g0=t_fit is not None,
        t_fit=t_fit, stopped_ep=stopped, acc=acc, h_z=h_z,
        wall=curve["wall"][-1], l_func10=l_func_at10(rec),
        ts=time.strftime("%Y-%m-%d %H:%M:%S")))
    print(f"  {cfg_tag(cfg_idx)} s{s} ep{rung_ep}: trained to ep{stopped} "
          f"acc={acc:.3f} H(Z_t)={h_z:.2f} T_fit={_fmt_t(t_fit)}", flush=True)
    return rec


def sh_score(cfg_idx, seeds, rung_ep, ckpt_root, mode="acc"):
    """Online pruning score (must stay isomorphic to score_at, otherwise the foundation self-check will raise an alarm).

    mode="acc"  (v1.0): (n_g0, median acc@rung, -median L_func@10)
    mode="time" (v1.1): (n_g0, -median epochs-to-gate (gated units only), median acc@rung)
    rung=ep units already early-stopped at the gate, so stopped is available online -- v1.1 needs no extra budget.
    """
    recs = [load_unit(cfg_idx, s, rung_ep, ckpt_root) for s in seeds]
    n_g0 = sum(1 for r in recs if r is not None and r["t_fit"] is not None)
    med_acc = statistics.median([r["acc"] for r in recs if r is not None])
    if mode in ("time", "time_norm"):
        rent = cost_rent(cfg_idx) if mode == "time_norm" else 1.0
        costs = [r["stopped"] * rent for r in recs
                 if r is not None and r["t_fit"] is not None]
        return (n_g0, -statistics.median(costs) if costs else -1e9, med_acc)
    return n_g0, med_acc, -statistics.median([l_func_at10(r) for r in recs
                                              if r is not None])


# ---------------------------------------------------------------------------
# --calibrate: axis B calibration (docstring (3))
# ---------------------------------------------------------------------------

def run_calibrate(seeds, resume, jsonl, ckpt_root):
    ep_full = RUNGS[-1]
    print("=" * 68)
    print(f"Axis B calibration: width in {WIDTHS} (euclid, lr=1e-3, {len(seeds)} seeds, "
          f"{ep_full}ep full curves)  [smoke={QUICK}]")
    print("Criterion reference: docstring (3) -- monotonicity test precedes any search data")
    print("=" * 68)
    for cfg_idx in CALIB_CFGS:
        for s in seeds:
            unit_train(cfg_idx, s, ep_full, resume, jsonl, ckpt_root, "calib")

    print("\n----- calibration analysis -----")
    med_l10, rows = {}, []
    for cfg_idx in CALIB_CFGS:
        cfg = CONFIGS[cfg_idx]
        recs = [load_unit(cfg_idx, s, ep_full, ckpt_root) for s in seeds]
        g0n = sum(1 for r in recs if r["t_fit"] is not None)
        tfits = [r["t_fit"] for r in recs if r["t_fit"] is not None]
        med_t = statistics.median(tfits) if tfits else None
        a10 = [r["curve"]["acc"][min(9, len(r["curve"]["acc"]) - 1)] for r in recs]
        l10 = [l_func_at10(r) for r in recs]
        med_l10[cfg["width"]] = statistics.median(l10)
        rows.append((cfg, g0n, med_t, a10, l10))
        print(f"  w{cfg['width']:>3}: G0 {g0n}/{len(seeds)}  "
              f"median T_fit={_fmt_t(med_t)}  "
              f"acc@10 min/med/max={min(a10):.3f}/{statistics.median(a10):.3f}/"
              f"{max(a10):.3f}  median L_func@10={med_l10[cfg['width']]:.2f}")

    print("\n(b) L_func@10 width monotonicity test (L(16) >= L(32) >= L(64) with at least one strict pair):")
    vals = [med_l10[w] for w in WIDTHS]
    noninc = all(vals[i] >= vals[i + 1] - 1e-12 for i in range(len(vals) - 1))
    strict = any(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
    if noninc and strict:
        print("  PASS -> SH pruning keeps the L_func@10 secondary tie-break signal")
        use_lfunc = True
    else:
        print(f"  FAIL (non-increasing={noninc}, strict pair={strict}) -> pruning falls back to pure acc@rung, "
              f"L_func recorded only, not used as a signal (pre-registered honest statement, recorded as-is)")
        use_lfunc = False

    print("\n(c) SH pruning reference lines (absolute thresholds for reporting only; SH actually uses ranks):")
    for cfg, _, _, a10, _ in rows:
        print(f"  w{cfg['width']:>3}: acc@10 three seeds = "
              f"{['%.3f' % a for a in sorted(a10)]}")
    print(f"\nConclusion: pruning signal definition = {'acc@rung + L_func@10 secondary' if use_lfunc else 'pure acc@rung'}")
    # persist the conclusion: read by --search when it runs independently (if missing, conservatively take the pure acc definition)
    with open(_calib_json(), "w", encoding="utf-8") as f:
        json.dump(dict(use_lfunc=use_lfunc, med_l10=med_l10,
                       ts=time.strftime("%Y-%m-%d %H:%M:%S")), f)
    return use_lfunc


# ---------------------------------------------------------------------------
# --search: Successive Halving (docstring (4))
# ---------------------------------------------------------------------------

def run_search(seeds, resume, jsonl, ckpt_root, use_lfunc=None, signal="time"):
    if use_lfunc is None:   # read the calibration conclusion (calibrate first, then search); if missing, conservatively take the pure acc definition
        use_lfunc = False
        if os.path.exists(_calib_json()):
            with open(_calib_json(), encoding="utf-8") as f:
                use_lfunc = bool(json.load(f).get("use_lfunc", False))
        else:
            print("[note] calibration conclusion not found (--calibrate not run) -> conservatively using the pure acc definition")
    if signal == "time_norm":
        sig_desc = "v1.2 epochs-to-gate x width normalization (n_g0, -median(epochs x width factor), median acc)"
    elif signal == "time":
        sig_desc = "v1.1 epochs-to-gate first (n_g0, -median epochs-to-gate, median acc)"
    else:
        sig_desc = ("v1.0 acc first (n_g0, median acc"
                    + (", -median L_func@10)" if use_lfunc else ", pure acc definition)"))
    print("=" * 68)
    print(f"SH search: {len(CONFIGS)} configs, eta={ETA}, rung={RUNGS}, seeds={seeds}  "
          f"[smoke={QUICK}]")
    print(f"Pruning signal = {sig_desc}")
    print("=" * 68)
    alive = list(range(len(CONFIGS)))
    finalists = []
    for r, ep in enumerate(RUNGS):
        n_keep = max(1, len(CONFIGS) // (ETA ** (r + 1)))
        print(f"\n--- rung{r + 1}: ep{ep} ({len(alive)} configs, keep {n_keep}) ---")
        for cfg_idx in alive:
            for s in seeds:
                unit_train(cfg_idx, s, ep, resume, jsonl, ckpt_root, "search")

        def key(c):
            s = sh_score(c, seeds, ep, ckpt_root, signal)
            if signal == "acc" and not use_lfunc:
                return (s[0], s[1], 0.0)   # calibration did not pass -> L_func is not used as a signal
            return s

        scored = sorted(alive, key=key, reverse=True)
        keep = scored[:n_keep]
        for c in scored:
            s = sh_score(c, seeds, ep, ckpt_root, signal)
            mark = "KEEP" if c in keep else "cut "
            if signal == "time":
                head = f"median epochs-to-gate={-s[1]:.1f}"
            elif signal == "time_norm":
                head = f"median normalized cost={-s[1]:.1f}"
            else:
                head = f"median acc={s[1]:.3f}"
            print(f"  {mark} {cfg_tag(c)}  G0={s[0]}/{len(seeds)} {head}")
        if r == len(RUNGS) - 1:
            finalists = keep
        else:
            alive = keep

    print(f"\n----- Final (rung3 survivors, ascending median epochs-to-gate = P5 definition) -----")
    rank_finalists(finalists, seeds, ckpt_root)
    print("Verdict reference: docstring (4) -- feasibility 2/3 rule; winner = smallest median epochs-to-gate.")


def load_deepest(cfg_idx, s, ckpt_root):
    """Deepest available rung unit reading (T_fit is the early-stop reading and is independent of rung depth;
    configs pruned by SH only have shallow-rung data, and report must be able to read them)."""
    for ep in reversed(RUNGS):
        rec = load_unit(cfg_idx, s, ep, ckpt_root)
        if rec is not None:
            return rec
    return None


def rank_finalists(cfg_list, seeds, ckpt_root):
    """Final ranking: primary definition = median epochs-to-gate (ruled by P5; the wall-clock T_fit serves only as a reference column).

    The wall clock is not comparable across runs (load contamination), so it is no longer a sorting key; T_fit only breaks ties when epoch counts are equal.
    """
    need = max(1, math.ceil(2 * len(seeds) / 3))   # feasibility 2/3 rule (docstring (4))
    ranked = []
    for c in cfg_list:
        recs = [load_deepest(c, s, ckpt_root) for s in seeds]
        if all(r is None for r in recs):
            print(f"  [no data] {cfg_tag(c)} (not run or not reached rung{RUNGS[-1]})")
            continue
        g0n = sum(1 for r in recs if r is not None and r["t_fit"] is not None)
        passed = [r for r in recs if r is not None and r["t_fit"] is not None]
        med_ep = statistics.median([r["stopped"] for r in passed]) if passed else None
        med_t = statistics.median([r["t_fit"] for r in passed]) if passed else None
        ranked.append((c, g0n, med_ep, med_t))
    feas = [x for x in ranked if x[2] is not None and x[1] >= need]
    infeas = [x for x in ranked if x[2] is None or x[1] < need]
    feas.sort(key=lambda x: (x[2], x[3]))
    for i, (c, g0n, med_ep, med_t) in enumerate(feas):
        frag = " [fragile]" if g0n < len(seeds) else ""
        print(f"  #{i + 1} {cfg_tag(c)}  median epochs={med_ep}  "
              f"median T_fit={med_t:.0f}s  G0={g0n}/{len(seeds)}{frag}")
    for c, g0n, _, _ in infeas:
        why = f" (fragile: only {g0n}/{len(seeds)} passed the gate, less than 2/3)" if g0n > 0 \
            else " (failed the gate within 40ep)"
        print(f"  infeasible {cfg_tag(c)}  G0={g0n}/{len(seeds)}{why}")
    return feas


def run_rho(winner, seeds, resume, jsonl, ckpt_root):
    """rho audit of the winner config (docstring (5)): teacher replay -> distillation grid -> K*/K_min."""
    ep_full = RUNGS[-1]
    print(f"\n----- rho audit: {cfg_tag(winner)} x {len(seeds)} seeds -----")
    budget = cell_budget("cnn", QUICK)
    n_pc = N_PER_CLASS_QUICK if QUICK else N_PER_CLASS
    rho_list = []
    for s in seeds:
        tck = load_unit(winner, s, ep_full, ckpt_root)
        if tck is None:
            print(f"  seed {s}: [no teacher data] skip")
            continue
        torch.manual_seed(teacher_seed(winner, s))
        (xtr, ytr), (xte, yte) = make_data(n_pc, s)
        train_loader = DataLoader(list(zip(xtr, ytr)), batch_size=128, shuffle=True)
        test_loader = DataLoader(list(zip(xte, yte)), batch_size=256)
        teacher = GridCNN(d=64, n_codes=CONFIGS[winner]["kt"],
                          width=CONFIGS[winner]["width"],
                          metric=CONFIGS[winner]["metric"])
        teacher.load_state_dict(tck["state_dict"])
        acc_t, hz_t = tck["acc"], tck["h_z"]
        ok, _, _ = g0_gate(acc_t, hz_t, CONFIGS[winner]["kt"])
        k_min = math.ceil(math.exp(task_info_fano(1 - acc_t, C_TASK)))
        print(f"  seed {s}: teacher acc={acc_t:.3f} H(Z_t)={hz_t:.2f} "
              f"G0={'PASS' if ok else 'FAIL'} K_min={k_min} (T_fit={_fmt_t(tck['t_fit'])})")

        accs = {}
        for K in CALIBRATED_K:
            torch.manual_seed(STUDENT_SEED_BASE + 100 * winner + 100 * s + K)
            student = GridCNN(d=64, n_codes=K, width=CONFIGS[winner]["width"],
                              metric=CONFIGS[winner]["metric"])
            spath = os.path.join(ckpt_root, f"cfg{winner}_student_s{s}_K{K}.pt")
            if resume and os.path.exists(spath):
                ck = torch.load(spath, map_location="cpu", weights_only=True)
                accs[K] = float(ck["acc"])
                print(f"    K={K}: [cache hit] acc={accs[K]:.3f}", flush=True)
                continue
            opt = torch.optim.Adam(student.parameters(), lr=budget["lr"])
            for _ in range(budget["epochs"]):
                train_one_epoch(student, train_loader, opt, distill_from=teacher)
            acc_s, _ = evaluate(student, test_loader)
            accs[K] = acc_s
            torch.save({"acc": acc_s}, spath)
            print(f"    K={K}: acc={acc_s:.3f}", flush=True)

        plateau = distill_plateau_ok(CALIBRATED_K, accs)
        k_star = distill_gap_k_star(acc_t, CALIBRATED_K, accs)
        rho = (k_star / k_min) if k_star is not None else None
        rho_list.append(rho)
        _append_jsonl(jsonl, dict(mode="rho_summary", cfg_idx=winner, seed=s,
                                  acc_t=acc_t, k_min=k_min, plateau_ok=plateau,
                                  k_star=k_star, rho=rho,
                                  ts=time.strftime("%Y-%m-%d %H:%M:%S")))
        ps = "plateau ok" if plateau else "no plateau"
        if rho is not None:
            print(f"    K*={k_star} ({ps}) rho={rho:.2f}")
        else:
            print(f"    K*=not located ({ps})")

    vals = [r for r in rho_list if r is not None]
    if vals:
        med = statistics.median(vals)
        verdict = ("non-pathological" if RHO_AUDIT_LO <= med <= RHO_AUDIT_HI else "pathological")
        print(f"\nAudit conclusion: median rho={med:.2f} ({len(vals)}/{len(seeds)} seeds located)"
              f" -> falls in [{RHO_AUDIT_LO}, {RHO_AUDIT_HI}] = {verdict}"
              f" (audit only, does not enter reward)")
    else:
        print("\nAudit conclusion: no seed located K* (plateau not established) -- recorded as-is")


# ---------------------------------------------------------------------------
# --report: summary + winner rho audit (docstring (4)(5))
# ---------------------------------------------------------------------------

def run_report(seeds, with_rho, resume, jsonl, ckpt_root):
    print("=" * 68)
    print(f"Report: {len(CONFIGS)} configs x {len(seeds)} seeds (deepest rung reading)  "
          f"[smoke={QUICK}]")
    print("Criterion reference: docstring (4) feasibility 2/3 rule; (5) rho audit interval "
          f"[{RHO_AUDIT_LO}, {RHO_AUDIT_HI}]")
    print("=" * 68)
    all_cfg = list(range(len(CONFIGS)))
    feas = rank_finalists(all_cfg, seeds, ckpt_root)
    if not feas:
        print("\nNo feasible configs (search may be incomplete) -- run --search first, then --report.")
        return
    winner = feas[0][0]
    print(f"\nWinner = {cfg_tag(winner)}")
    if not with_rho:
        print("(--rho not requested, winner audit skipped)")
        return
    run_rho(winner, seeds, resume, jsonl, ckpt_root)


# ---------------------------------------------------------------------------
# --main: optimizer-effectiveness main experiment (docstring (8))
# ---------------------------------------------------------------------------

DEFAULT_CFG = 13          # GridCNN default definition: metric=cosine, width=32, lr=1e-3
RS_SEED = 20260923        # fixed seed for random search (reproducible)
RS_TRIALS = 200           # number of independent trials for the equal-budget random search


def acc_at(rec, ep):
    """The acc of the unit curve at epoch ep (bit-for-bit replicating rec["acc"] of an online rung=ep unit)."""
    c = rec["curve"]["acc"]
    return c[min(ep, len(c)) - 1]


def g0_within(rec, ep):
    """Whether the unit passes G0 before epoch ep (equivalent to t_fit is not None of an online rung=ep unit)."""
    return rec["t_fit"] is not None and rec["stopped"] <= ep


def score_at(cfg_idx, seeds, ep, ckpt_root, use_g0=True, use_lfunc=False,
             mode="acc"):
    """Offline replay pruning score (equivalence argument see docstring (8)); mode selects the signal version.

    mode="acc"  (v1.0): (n_g0, median acc@rung, -median L_func@10)
    mode="time" (v1.1): (n_g0, -median epochs-to-gate (gated units only), median acc@rung)
        -- fewer epochs-to-gate ranks first; non-gated ones are automatically placed after by n_g0 (a placeholder -1e9 when there are no gated units,
        in which case all are tied and the third term acc orders them).
    """
    recs = [r for r in (load_unit(cfg_idx, s, RUNGS[-1], ckpt_root)
                        for s in seeds) if r is not None]
    if not recs:
        return (0, 0.0, 0.0)
    n_g0 = sum(1 for r in recs if g0_within(r, ep))
    ng = n_g0 if use_g0 else 0
    med_acc = statistics.median([acc_at(r, ep) for r in recs])
    if mode in ("time", "time_norm"):
        rent = cost_rent(cfg_idx) if mode == "time_norm" else 1.0
        costs = [r["stopped"] * rent for r in recs if g0_within(r, ep)]
        return (ng, -statistics.median(costs) if costs else -1e9, med_acc)
    neg_l = -statistics.median([l_func_at10(r) for r in recs]) if use_lfunc else 0.0
    return (ng, med_acc, neg_l)


def replay_sh(seeds, ckpt_root, use_g0=True, use_lfunc=False, eta=ETA,
              mode="acc", pool=None):
    """Offline replay of the SH protocol, returning (final chosen config, per-rung trace).

    pool: the config indices participating in the search (all by default). For v2's layered analysis, pass a subset by K_t layer --
    different K_t layers have different G0 gates, so cross-layer comparison does not hold (docstring (11) Q5).
    """
    pool = list(range(len(CONFIGS))) if pool is None else list(pool)
    n_cfg = len(pool)
    alive = list(pool)
    trace = []
    for r, ep in enumerate(RUNGS):
        n_keep = max(1, n_cfg // (eta ** (r + 1)))
        scored = sorted(alive, key=lambda c: score_at(c, seeds, ep, ckpt_root,
                                                      use_g0, use_lfunc, mode),
                        reverse=True)
        keep = scored[:n_keep]
        trace.append(dict(ep=ep, ranked=scored, keep=keep))
        alive = keep
    return alive[0], trace


def sh_nominal_cost(seeds, eta=ETA, pool=None):
    """Nominal cost (epoch units): Σ over rungs of surviving config count x seed count x rung epoch count.

    pool must match the config set actually participating in the search (use collect_pool() when --layers limits layers),
    otherwise the nominal budget is computed over 90 configs while only 54 are actually evaluated, distorting the equal-budget comparison against random search.
    """
    n_cfg = len(collect_pool()) if pool is None else len(pool)
    alive = n_cfg
    total = 0
    for r, ep in enumerate(RUNGS):
        total += alive * len(seeds) * ep
        alive = max(1, n_cfg // (eta ** (r + 1)))
    return total


def gold_table(seeds, ckpt_root):
    """Ground-truth table: per-config median T_fit / median stopped_ep / G0 pass count (docstring (8))."""
    out = {}
    for c in range(len(CONFIGS)):
        recs = [r for r in (load_unit(c, s, RUNGS[-1], ckpt_root)
                            for s in seeds) if r is not None]
        tfits = [r["t_fit"] for r in recs if r["t_fit"] is not None]
        stops = [r["stopped"] for r in recs if r["t_fit"] is not None]
        out[c] = dict(n_g0=len(tfits),
                      t_fit=statistics.median(tfits) if tfits else None,
                      stopped=statistics.median(stops) if stops else None)
    return out


def evaluate_scheme(chosen, tabs, need):
    """Score a scheme: (median T_fit, whether feasible)."""
    if chosen is None:
        return None, False
    t = tabs[chosen]["t_fit"]
    if t is None or tabs[chosen]["n_g0"] < need:
        return None, False
    return t, True


def kt_layers():
    """The config index interval of each K_t layer (for v2's layered analysis; docstring (11) Q5)."""
    n = len(METRICS) * len(WIDTHS) * len(LRS)
    return [(kt, list(range(i * n, (i + 1) * n)))
            for i, kt in enumerate(K_TEACHERS)]


def collect_pool():
    """The config indices **actually collected** this round (narrowed when --layers limits layers; otherwise all).

    [2026-09-25 harness fix] Phase 2 replay originally used pool=None (= all 90 configs), while
    --layers collected only 54 -> n_keep = 90//3^(r+1) computed over 90, while only 54 are actually evaluable,
    effectively loosening the η=3 pruning ratio to 54/30≈1.8; uncollected layers would also be
    "pruned for free" as (0, 0.0, 0.0). So the replay pool / nominal budget / random-search sampling are all changed to follow this function.
    """
    if COLLECT_KT:
        return [c for c in range(len(CONFIGS)) if CONFIGS[c]["kt"] in COLLECT_KT]
    return list(range(len(CONFIGS)))


def layer_units(idxs, seeds, ckpt_root):
    """The number of units that actually exist in this layer (0 = not collected; a decision must be recorded as 'undecidable', not 'infeasible')."""
    return sum(1 for c in idxs for s in seeds
               if load_unit(c, s, RUNGS[-1], ckpt_root) is not None)


def metric_interaction(idxs, feas, seeds, ckpt_root):
    """Whether this layer exhibits a "feasible all cosine / infeasible all euclid" codebook x metric interaction (T3).

    Returns None = the layer was not collected (undecidable); otherwise dict(cos=(feasible, total), euc=..., perfect=bool).
    """
    if layer_units(idxs, seeds, ckpt_root) == 0:
        return None
    tab = {}
    for c in idxs:
        f, n = tab.get(CONFIGS[c]["metric"], (0, 0))
        tab[CONFIGS[c]["metric"]] = (f + int(c in feas), n + 1)
    cos_f, cos_n = tab.get("cosine", (0, 0))
    euc_f, euc_n = tab.get("euclid", (0, 0))
    return dict(cos=(cos_f, cos_n), euc=(euc_f, euc_n),
                perfect=(cos_n > 0 and euc_n > 0 and cos_f == cos_n and euc_f == 0))


def spearman(xs, ys):
    """Spearman rank correlation (no scipy dependency; ties get the average rank)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


def width_flops_factor(width):
    """Width cost factor (a purely architectural quantity, the noise-free width rent; used by P9).

    GridCNN compute approximation: conv1(1->w,3x3,16x16) + conv2(w->2w,3x3,8x8)
    + conv3(2w->2w,3x3,4x4) + fc1(4w->64) -- counted as multiply-add, giving 1728*w^2 + 2560*w.
    After normalization 16:32:64 is about 1 : 3.83 : 15.0 (superlinear, dominated by the w^2 terms of conv2/conv3).
    """
    return 1728 * width * width + 2560 * width


def per_epoch_flops(cfg_idx):
    """Per-epoch per-sample compute (multiply-add): convolution term + codebook lookup term K·d (docstring (12))."""
    c = CONFIGS[cfg_idx]
    return width_flops_factor(c["width"]) + VQ_D * c["kt"]


def cost_rent(cfg_idx, with_kt=True):
    """Cost rent multiplier (a common denominator = the smallest (width, K_t) config, so variants are directly comparable).

    with_kt=True : convolution + codebook lookup K·d -- shared by the v1.2 signal / P9 / the primary definition
    with_kt=False: convolution term only -- the V1 cost-term sensitivity comparison (docstring (12))
    """
    c = CONFIGS[cfg_idx]
    ref = per_epoch_flops(CFG_INDEX[("euclid", WIDTHS[0], LRS[0],
                                     min(K_TEACHERS))])
    num = width_flops_factor(c["width"])
    if with_kt:
        num += VQ_D * c["kt"]
    return num / ref


def holdout_test(sel_seeds, ev_seeds, ckpt_root, modes=("acc", "time"),
                 metric="ep"):
    """Holdout-seed generalization test (P8): sel_seeds for selection -> ev_seeds for evaluation.

    Non-triviality: when the signal is isomorphic to the criterion, hitting it on the same batch of seeds is definitionally necessary; only
    whether it still hits after switching to another batch of data seeds is informative. The evaluation definition is decided by metric ("ep" = median epochs-to-gate;
    "ep_norm" = additionally multiplied by the width rent, isomorphic to the v1.2 signal/P9 definition).
    Returns {mode: dict(cfg, val, gold, gold_val, regret)}; None when the holdout set has no feasible config.
    """
    def med_val(c, sd):
        """Holdout-set evaluation value: metric="ep" uses the median epochs-to-gate, "ep_norm" multiplies by the width rent."""
        v = [r["stopped"] for r in (load_unit(c, s, RUNGS[-1], ckpt_root)
                                    for s in sd)
             if r is not None and r["t_fit"] is not None]
        if not v:
            return None
        m = statistics.median(v)
        return m * (cost_rent(c) if metric == "ep_norm" else 1.0)

    need_e = max(1, math.ceil(2 * len(ev_seeds) / 3))
    POOL = collect_pool()
    ng = {}
    for c in POOL:
        n = 0
        for s in ev_seeds:
            r = load_unit(c, s, RUNGS[-1], ckpt_root)
            n += int(r is not None and r["t_fit"] is not None)
        ng[c] = n
    ev = {c: med_val(c, ev_seeds) for c in POOL}
    feas_e = {c: v for c, v in ev.items() if v is not None and ng[c] >= need_e}
    if not feas_e:
        return None
    gold_e = min(feas_e, key=lambda c: feas_e[c])
    out = {}
    for m in modes:
        w = replay_sh(sel_seeds, ckpt_root, mode=m, pool=POOL)[0]
        out[m] = dict(cfg=w, val=feas_e.get(w), gold=gold_e,
                      gold_val=feas_e[gold_e],
                      regret=(feas_e[w] - feas_e[gold_e]) if w in feas_e else None)
    return out


def verify_replay(seeds, ckpt_root):
    """Foundation self-check: does the pruning score replayed offline from ep40 units replicate the online score bit-for-bit?

    The online score comes from the ep10/ep20 units really produced during the search stage (sh_score), and the offline score comes from the
    ep40 units of the main experiment's Phase 1 (score_at). The two should be bit-for-bit identical (argument see docstring (8)) --
    this is the premise of the whole offline replay design and must be measured rather than assumed.

    Only **informative** points are counted: (a) the ep40 unit of that (config, seed) must be genuinely trained,
    since a promotion-copy record is by construction equal to the ep10 record and the comparison is trivially true, so it is not counted; (b) that seed must have an online
    ep10/ep20 unit to compare against (the search stage only ran seeds 0-2, so the check actually covers seeds 0-2).
    The first two terms (n_g0, median acc) are compared; the third term L_func is always computed on the online side and is switch-controlled on the offline side,
    so it is not compared.
    """
    bad = checked = skipped = 0
    for c in range(len(CONFIGS)):
        for ep in RUNGS[:-1]:
            sd = []
            for s in seeds:
                u40 = load_unit(c, s, RUNGS[-1], ckpt_root)
                if u40 is None or u40.get("promoted_from") is not None:
                    continue          # no genuine ep40, or it is a promotion copy (trivially true, not counted)
                if load_unit(c, s, ep, ckpt_root) is None:
                    continue          # no online control
                sd.append(s)
            if len(sd) < 2:
                skipped += 1
                continue
            on = tuple(sh_score(c, sd, ep, ckpt_root, "acc")[:2])
            off = tuple(score_at(c, sd, ep, ckpt_root, mode="acc")[:2])
            checked += 1
            if on != off:
                bad += 1
                print(f"    [MISMATCH] cfg{c} ep{ep} seeds{sd}: online={on} offline={off}")
    if checked == 0:
        print(f"  foundation selftest: no valid comparison points (skipped {skipped} (config, rung))")
    else:
        print(f"  foundation selftest: {checked - bad}/{checked} valid comparison points bit-for-bit identical"
              f" (skipped {skipped} points with no genuine ep40 / no online control)"
              f"{'  PASS' if bad == 0 else '  MISMATCHES FOUND, investigate before trusting Phase 2'}")
    return bad


def run_main(seeds, resume, jsonl, ckpt_root, replay_only=False):
    print("=" * 68)
    print(f"Main experiment: optimizer effectiveness ({len(CONFIGS)} configs x {len(seeds)} seeds, "
          f"exhaustive gold + offline replay)  [smoke={QUICK}]"
          + ("  [replay-only, skipping Phase 1]" if replay_only else ""))
    print("Criterion reference: docstring (8)(9)(10)(11)(13); honest statement: v1 data already seen, new layers are fresh data")
    print("=" * 68)

    # ---- Phase 1: exhaustive (the only compute-consuming step; skipped when replay_only) ----
    if replay_only:
        print(f"\n--- Phase 1 skipped (--replay-only): reading existing checkpoints only ---")
    else:
        print(f"\n--- Phase 1: exhaustive collection ({len(CONFIGS) * len(seeds)} units, early stop at "
              f"{RUNGS[-1]}ep) ---")
        for c in range(len(CONFIGS)):
            if COLLECT_KT and CONFIGS[c]["kt"] not in COLLECT_KT:
                continue          # --layers layer limiting (used for the reduced-space generalization test)
            for s in seeds:
                unit_train(c, s, RUNGS[-1], resume, jsonl, ckpt_root, "main")
        verify_replay(seeds, ckpt_root)
    main_report(seeds, jsonl, ckpt_root)


def main_report(seeds, jsonl, ckpt_root):
    """Phase 2: offline replay comparison and criterion decisions from the gold table (zero training, callable independently)."""
    tabs = gold_table(seeds, ckpt_root)
    need = max(1, math.ceil(2 * len(seeds) / 3))
    feas = {c: v for c, v in tabs.items()
            if v["t_fit"] is not None and v["n_g0"] >= need}
    if not feas:
        print("\nNo feasible configs -- main experiment UNDECIDABLE, recorded as-is.")
        return
    # primary definition (docstring (12) V2): normalized cost = epochs-to-gate x cost rent (including the codebook term K·d)
    # the wall-clock T_fit is still kept as a control (P5: not comparable across runs); the conv-only rent variant serves the V1 sensitivity comparison
    def cost_of(c, with_kt=True):
        return feas[c]["stopped"] * cost_rent(c, with_kt)

    gold = min(feas, key=lambda c: (cost_of(c), feas[c]["t_fit"]))
    gold_conv = min(feas, key=lambda c: (cost_of(c, False), feas[c]["t_fit"]))
    gold_w = min(feas, key=lambda c: feas[c]["t_fit"])
    gold_t = feas[gold]["t_fit"]
    gold_ep = feas[gold]["stopped"]

    # ---- gold ground-truth table ----
    print(f"\n--- gold ground-truth table (ascending normalized cost = primary definition (incl. codebook term); feasibility "
          f"{need}/{len(seeds)} rule) ---")
    order = sorted(tabs, key=lambda c: (c not in feas,
                                        cost_of(c) if c in feas else 0.0,
                                        tabs[c]["t_fit"] or 1e9))
    for i, c in enumerate(order):
        v = tabs[c]
        flag = " <== GOLD" if c == gold else ""
        fat = " [fragile]" if 0 < v["n_g0"] < len(seeds) and v["t_fit"] is not None else ""
        bad = " infeasible" if v["n_g0"] < need else ""
        ctxt = f"{cost_of(c):.1f}" if c in feas else "None"
        print(f"  #{i + 1:>2} {cfg_tag(c)}  median epochs={v['stopped']}  "
              f"norm cost={ctxt}  median T_fit={_fmt_t(v['t_fit'])}  "
              f"G0={v['n_g0']}/{len(seeds)}{fat}{bad}{flag}")

    # P5: wall-clock vs epoch-count ranking consistency + two-definition gold comparison
    # note: order is already sorted by the primary definition (normalized cost), so consistency must be checked along the **wall-clock order** (otherwise it is trivially true)
    by_wall = sorted((c for c in tabs if tabs[c]["t_fit"] is not None),
                     key=lambda c: tabs[c]["t_fit"])
    mono = all(tabs[by_wall[i]]["stopped"] <= tabs[by_wall[i + 1]]["stopped"]
               for i in range(len(by_wall) - 1))
    print(f"\nP5 noise control: wall-clock order and epoch order {'consistent' if mono else 'inconsistent (wall-clock jitter)'}; "
          f"wall-clock-definition gold={cfg_tag(gold_w)} vs primary-definition gold={cfg_tag(gold)} -> "
          f"{'definitions agree' if gold_w == gold else 'definitions differ, wall-clock definition untrustworthy'}")

    # ---- Phase 2: offline replay ----
    print("\n--- Phase 2: offline replay comparison (zero extra training) ---")
    POOL = collect_pool()          # narrowed when layers are limited (see collect_pool docstring)
    if len(POOL) != len(CONFIGS):
        print(f"  [RESTRICTED] this round collected only {len(POOL)}/{len(CONFIGS)} configs (K_t="
              f"{sorted(COLLECT_KT)}); replay pool / nominal budget / random-search sampling all narrowed accordingly")
    rows = []
    # precompute the rung1 score once (rs needs thousands of queries, so it cannot read from disk every time)
    s10 = {c: score_at(c, seeds, 10, ckpt_root)[:2] for c in POOL}

    def add(name, w, cost):
        """Register a scheme: the primary regret uses normalized cost (primary definition, docstring (12) V2),
        while the epoch-count/wall-clock differences serve only as reference columns."""
        t, ok = evaluate_scheme(w, tabs, need)
        rows.append(dict(
            name=name, cfg=w, t_fit=t, cost=cost, hit=(w == gold),
            r_cost=(cost_of(w) - cost_of(gold)) if ok else None,
            r_ep=(tabs[w]["stopped"] - gold_ep) if ok else None,
            r_wall=(t - gold_t) if ok else None))

    add("exhaust(ground-truth upper bound)", gold, len(POOL) * len(seeds) * RUNGS[-1])

    # SH variants (v1.0 acc / v1.1 epochs-to-gate / v1.2 epochs-to-gate x width normalization)
    sh_specs = [("sh8", seeds, True, "acc"), ("sh3", seeds[:3], True, "acc"),
                ("sh8_noG0", seeds, False, "acc"),
                ("sh8_time", seeds, True, "time"),
                ("sh8_timenorm", seeds, True, "time_norm")]
    sh_traces = {}
    for name, sd, use_g0, mode in sh_specs:
        w, trace = replay_sh(sd, ckpt_root, use_g0, mode=mode, pool=POOL)
        sh_traces[name] = trace
        add(name, w, sh_nominal_cost(sd, pool=POOL))
        # P4: gold survival
        surv = ["survived" if gold in tr["keep"] else "pruned" for tr in trace]
        print(f"  {name}: final pick {cfg_tag(w)}  gold status at each rung={surv}")

    # sf10 single fidelity
    add("sf10(single fidelity)", max(POOL, key=lambda c: s10[c]),
        len(POOL) * len(seeds) * 10)

    # rs equal-budget random search (equal nominal cost; after sampling, pick one by the rung1 score)
    sh8_cost = sh_nominal_cost(seeds, pool=POOL)
    n_draw = max(1, sh8_cost // (len(seeds) * 10))
    rng = random.Random(RS_SEED)
    rs_r, rs_hit = [], 0
    for _ in range(RS_TRIALS):
        pick = max((POOL[rng.randrange(len(POOL))] for _ in range(n_draw)),
                   key=lambda c: s10[c])
        _, ok = evaluate_scheme(pick, tabs, need)
        rs_r.append((cost_of(pick) - cost_of(gold)) if ok else None)
        rs_hit += int(pick == gold)
    fin = [r for r in rs_r if r is not None]
    med_rs = statistics.median(fin) if fin else None
    inf_rate = 1.0 - len(fin) / RS_TRIALS
    rtxt = "None" if med_rs is None else f"{med_rs:.1f}"
    print(f"\n  rs(equal-budget random search, {n_draw} draws/trial, {RS_TRIALS} trials): "
          f"hit gold {rs_hit}/{RS_TRIALS} ({100.0 * rs_hit / RS_TRIALS:.1f}%)  "
          f"median regret(cost)={rtxt}  infeasible rate={100.0 * inf_rate:.1f}%")

    # default default definition
    add("default(zero search)", DEFAULT_CFG, 0)

    # ---- comparison table ----
    print(f"\n--- comparison table (primary definition = normalized cost (incl. codebook term): gold={cfg_tag(gold)} cost="
          f"{cost_of(gold):.2f} epochs={gold_ep}) ---")
    print(f"  {'scheme':<18}{'picked config':<34}{'epochs':>6}{'norm cost':>10}"
          f"{'regret(cost)':>13}{'regret(ep)':>11}{'nominal cost':>10}  hit gold")
    for r in rows:
        ept = str(tabs[r["cfg"]]["stopped"]) if r["t_fit"] is not None else "infeasible"
        cst = "-" if r["t_fit"] is None else f"{cost_of(r['cfg']):.1f}"
        rco = "-" if r["r_cost"] is None else f"{r['r_cost']:g}"
        rep = "-" if r["r_ep"] is None else f"{r['r_ep']:g}"
        print(f"  {r['name']:<18}{cfg_tag(r['cfg']):<34}{ept:>6}{cst:>10}"
              f"{rco:>13}{rep:>11}{r['cost']:>10}  {'yes' if r['hit'] else 'no'}")

    # ---- P8/P13: holdout-seed generalization (select on the first 3 seeds -> evaluate on the last 5 seeds) ----
    ho = ho_n = None
    if len(seeds) >= 6:
        ho = holdout_test(seeds[:3], seeds[3:], ckpt_root, metric="ep")
        ho_n = holdout_test(seeds[:3], seeds[3:], ckpt_root,
                            modes=("acc", "time", "time_norm"), metric="ep_norm")
    if ho:
        print(f"\n--- P8/P13 holdout generalization (select seeds {seeds[:3]} -> evaluate seeds {seeds[3:]}) ---")
        for tag, hh in (("def=epochs", ho), ("def=normalized cost", ho_n)):
            gv = "-" if hh["acc"]["gold_val"] is None else f"{hh['acc']['gold_val']:.1f}"
            print(f"  [{tag}] holdout-set gold = {cfg_tag(hh['acc']['gold'])}  value={gv}")
            for m, d in hh.items():
                rt = "infeasible" if d["regret"] is None else f"{d['regret']:g}"
                vv = "-" if d["val"] is None else f"{d['val']:.1f}"
                print(f"    signal {m:>9}: picked {cfg_tag(d['cfg'])}  value={vv}  regret={rt}")
    else:
        print("\nP8/P13 holdout generalization: insufficient seeds (need >=6) or no feasible config in holdout set -- skipped")

    # ---- V1: cost-term verification (docstring (12)) ----
    cost_ep = {c: cost_of(c) for c in feas}
    print(f"\n--- V1 cost-term verification: count the codebook lookup term K·d (d={VQ_D}) into the rent ---")
    for w in WIDTHS:
        rs = {kt: cost_rent(CFG_INDEX[("euclid", w, LRS[0], kt)])
              for kt in K_TEACHERS}
        pen = rs[512] / rs[256] - 1.0
        print(f"  w{w:>2}: rent K128={rs[128]:.4f}  K256={rs[256]:.4f}  "
              f"K512={rs[512]:.4f}  -> 512 vs 256 penalty {pen * 100:+.1f}%")
    i256 = CFG_INDEX[("euclid", 16, 3e-3, 256)]
    i512 = CFG_INDEX[("euclid", 16, 3e-3, 512)]
    if i256 in feas and i512 in feas:
        e256, e512 = feas[i256]["stopped"], feas[i512]["stopped"]
        pen = cost_rent(i512) / cost_rent(i256) - 1.0
        print(f"  measured on the same triple (euclid/w16/lr3e-3): epochs {e256} -> {e512} "
              f"({e512 / e256 - 1.0:+.0%}), while the rent penalty is only {pen * 100:+.1f}%")
    print(f"  gold(conv-only rent)={cfg_tag(gold_conv)}  vs  "
          f"gold(incl. VQ)={cfg_tag(gold)} -> "
          f"{'choice unchanged (penalty insufficient, SUPPORTS scheme 3)' if gold_conv == gold else 'choice changed (recorded as-is)'}")
    print("  Top 3 by primary definition (incl. VQ):")
    for c in sorted(feas, key=lambda c: (cost_ep[c], feas[c]["t_fit"]))[:3]:
        print(f"    {cfg_tag(c)}  epochs={feas[c]['stopped']}  "
              f"norm cost={cost_ep[c]:.2f}")

    # ---- v2 layered analysis (docstring (11) Q1/Q2/Q5) ----
    # intra-layer comparison is valid (the same layer has the same G0 gate); cross-layer "epochs-to-gate" cannot be compared directly -- note it honestly
    print("\n--- v2 layered analysis (K_t axis; comparable within a layer, not directly comparable across layers due to different gates) ---")
    layer_best = {}
    for kt, idxs in kt_layers():
        feas_l = [c for c in idxs if c in feas]
        ng_tot = sum(tabs[c]["n_g0"] for c in idxs)
        n_units = sum(1 for c in idxs for s in seeds
                      if load_unit(c, s, RUNGS[-1], ckpt_root) is not None)
        nodata = "  [no data]" if n_units == 0 else ""
        if feas_l:
            b = min(feas_l, key=lambda c: (cost_ep[c], feas[c]["t_fit"]))
            layer_best[kt] = b
            btxt = f"{cfg_tag(b)}  norm cost={cost_ep[b]:.1f}  epochs={feas[b]['stopped']}"
        else:
            layer_best[kt] = None
            btxt = "no feasible config"
        print(f"  K_t={kt:>4} (gate H(Z_t)>={0.8 * math.log(kt):.2f}): "
              f"feasible {len(feas_l)}/{len(idxs)}  intra-layer optimum = {btxt}{nodata}")

    # ---- criterion decisions ----
    def regret_of(name):
        return next((r["r_cost"] for r in rows if r["name"] == name), None)

    print("\n--- pre-registered criterion verdicts (docstring (8) and (9)) ---")
    r_sh8, r_sh3 = regret_of("sh8"), regret_of("sh3")
    r_noG0 = regret_of("sh8_noG0")
    r_sf = regret_of("sf10(single fidelity)")
    if r_sh8 is None or r_sh3 is None:
        p1 = "UNDECIDABLE"
    elif r_sh8 < r_sh3:
        p1 = "SUPPORT (8 seeds better)"
    elif r_sh8 == r_sh3:
        p1 = "SUPPORT but tied (seed count does not change the choice)"
    else:
        p1 = "NO SUPPORT (more seeds were worse, recorded as-is)"
    print(f"  P1 seed-count effect: regret(sh8)={r_sh8} vs regret(sh3)={r_sh3} -> {p1}")
    v2 = (r_sh8 is not None and med_rs is not None and r_sh8 < med_rs)
    print(f"  P2 pruning value: regret(sh8)={r_sh8} < random-search median={med_rs} -> "
          f"{'SUPPORT' if v2 else 'NO SUPPORT (pruning value insufficient, key negative result archived)'}")
    v3 = (r_sh8 is not None and r_noG0 is not None and r_sh8 <= r_noG0)
    print(f"  P3 hard-constraint necessity: regret(sh8)={r_sh8} <= regret(sh8_noG0)={r_noG0} -> "
          f"{'SUPPORT' if v3 else 'NO SUPPORT (L0 contributes nothing to selection)'}")
    cut = [tr["ep"] for tr in sh_traces["sh8"] if gold not in tr["keep"]]
    print(f"  P4 pruning collateral damage: gold pruned at rung(ep)={cut if cut else 'never pruned'} -> "
          f"{'collateral damage' if cut else 'no collateral damage'}")
    print(f"  P5 noise control: primary definition switched to normalized cost; two-definition gold "
          f"{'agree' if gold_w == gold else 'differ (wall-clock definition untrustworthy)'}")
    if r_sf is None or r_sh8 is None:
        p6 = "UNDECIDABLE"
    elif r_sf > r_sh8:
        p6 = "multi-fidelity has value"
    elif r_sf == r_sh8:
        p6 = "tied (multi-fidelity adds nothing)"
    else:
        p6 = "single fidelity suffices (multi-fidelity is worse)"
    print(f"  P6 multi-fidelity necessity: regret(sf10)={r_sf} vs regret(sh8)={r_sh8} -> {p6}")

    # ---- v1.1 pruning-signal verification (docstring (9)) ----
    r_time = regret_of("sh8_time")
    w_time = next(r["cfg"] for r in rows if r["name"] == "sh8_time")
    print(f"  P7 linkage: sh8_time final pick={cfg_tag(w_time)} regret={r_time} -> "
          f"{'hits gold (tautological, linkage check only)' if r_time == 0 else 'misses'}")
    if ho_n:
        r_tn_h = ho_n["time_norm"]["regret"]
        r_acc_h = ho_n["acc"]["regret"]
        ok8 = (r_tn_h == 0 and (r_acc_h is None or r_acc_h > 0))
        print(f"  P8 holdout generalization (normalized def = primary def): time_norm regret={r_tn_h} vs "
              f"acc regret={r_acc_h} -> "
              f"{'SUPPORT (normalized signal generalizes, acc does not)' if ok8 else 'NO SUPPORT (recorded as-is)'}")
    else:
        ok8 = False
        print("  P8 holdout generalization: UNDECIDABLE")
    ok9 = (gold_conv == gold)
    print(f"  P9 cost-term sensitivity (V1): gold(conv-only)={cfg_tag(gold_conv)} vs "
          f"gold(incl. VQ)={cfg_tag(gold)} -> "
          f"{'choice unchanged within the layer (penalty small at low K_t; penalty grows linearly with K_t, see R3)' if ok9 else 'choice changed (recorded as-is)'}")
    print(f"  P10 definition consistency: "
          f"{'v1.1 can replace v1.0 in production' if (ok8 and ok9) else 'v1.1 remains a candidate, v1.0 retained'}")

    # ---- v1.2 pruning-signal verification (docstring (10)) ----
    r_tn = regret_of("sh8_timenorm")
    w_tn = next(r["cfg"] for r in rows if r["name"] == "sh8_timenorm")
    print(f"  P11 linkage: sh8_timenorm final pick={cfg_tag(w_tn)} regret={r_tn} -> "
          f"{'hits gold (tautological, linkage check only)' if r_tn == 0 else 'misses'}")
    # P12 is evaluated within the K_t=256 layer (the v1 original space), keeping it comparable with the docstring (10) prediction
    v1_pool = kt_layers()[0][1]
    _, tr_t_v1 = replay_sh(seeds, ckpt_root, mode="time", pool=v1_pool)
    _, tr_n_v1 = replay_sh(seeds, ckpt_root, mode="time_norm", pool=v1_pool)
    r2_t = tr_t_v1[1]["keep"]
    r2_n = tr_n_v1[1]["keep"]
    print(f"  P12 rung2 survivors (K_t=256 layer): v1.1={[cfg_tag(c) for c in r2_t]}")
    print(f"                              v1.2={[cfg_tag(c) for c in r2_n]}")
    ok12 = bool(r2_n) and all(CONFIGS[c]["width"] == WIDTHS[0] for c in r2_n)
    print(f"      -> {'SUPPORT (wide configs blocked, signal aligned with true cost)' if ok12 else 'NO SUPPORT (width normalization had no effect, recorded as-is)'}")
    if ho_n:
        ok13 = (ho_n["time_norm"]["regret"] == 0)
        print(f"  P13 holdout generalization (normalized def): time_norm regret="
              f"{ho_n['time_norm']['regret']} -> {'SUPPORT' if ok13 else 'NO SUPPORT (recorded as-is)'}")
    else:
        ok13 = False
        print("  P13 holdout generalization: UNDECIDABLE")
    print(f"  P14 conclusion stability: winner {cfg_tag(w_tn)} vs gold {cfg_tag(gold)} -> "
          f"{'consistent (v1.2 only tightens the boundary, does not change the champion)' if w_tn == gold else 'inconsistent'}")
    print(f"  P15 production decision: "
          f"{'v1.2 replaces v1.1 (signal / P5 primary definition / P9 cost definition all aligned)' if (ok12 and ok13) else 'v1.2 remains a candidate, v1.1 retained'}")

    # ---- v2 criteria (docstring (11)(13): the K_t axis) ----
    layers = kt_layers()
    n_per = len(layers[0][1])
    kt_top = max(K_TEACHERS)
    f1 = [c for c in layers[1][1] if c in feas]          # K_t=128 (lower boundary)
    f_top = [c for c in layers[-1][1] if c in feas]      # K_t=max (upper boundary)
    n1 = layer_units(layers[1][1], seeds, ckpt_root)     # [2026-09-25] data guard
    if n1 == 0:
        print(f"  Q1 lower boundary of feasible region: K_t=128 layer **no data** (not collected) -- UNDECIDABLE"
              f" (must not read 'not collected' as 'infeasible')")
    else:
        print(f"  Q1 lower boundary of feasible region: K_t=128 layer feasible {len(f1)}/{n_per}"
              f" (criterion: <=2 counts as 'almost entirely infeasible') -> "
              f"{'SUPPORT (lower end of codebook axis is a hard boundary)' if len(f1) <= 2 else 'NO SUPPORT (the layer has feasible configs, boundary lies lower)'}")
    pick_kt = CONFIGS[w_tn]["kt"]
    b512 = layer_best.get(512)
    print(f"  Q2 greedy resistance: global winner layer K_t={pick_kt}; K_t=512 layer feasible "
          f"{len([c for c in layers[2][1] if c in feas])}/{n_per}"
          f" (intra-layer optimum {cfg_tag(b512) if b512 is not None else 'none'})")
    if b512 is not None and pick_kt == 512:
        print("      -> signal took 512: cost definition does not resist greed (recorded as-is)")
    else:
        print("      -> signal did not take 512: see group G (upper-bound layer behavior)")
    # ---- G1/G2/G3: K_t=max greedy-upper-bound measurement (docstring (13)) ----
    ng_top = sum(tabs[c]["n_g0"] for c in layers[-1][1])
    b_top = layer_best.get(kt_top)
    n_units_top = sum(1 for c in layers[-1][1] for s in seeds
                      if load_unit(c, s, RUNGS[-1], ckpt_root) is not None)
    if n_units_top == 0:
        print(f"  G1 upper-bound feasibility: K_t={kt_top} layer **no data** (not collected) -- UNDECIDABLE")
    else:
        print(f"  G1 upper-bound feasibility: K_t={kt_top} (codebook gate {0.8 * math.log(kt_top):.2f}) feasible "
              f"{len(f_top)}/{n_per}  G0 pass rate {ng_top}/{n_per * len(seeds)} -> "
              f"{'gate bites (greed has a natural upper bound)' if len(f_top) < n_per else 'gate still does not bite (upper bound must rely on artificial caps)'}")
    if b_top is not None and b512 is not None:
        c_top, c512 = cost_ep[b_top], cost_ep[b512]
        print(f"  G2 greedy payoff: upper-bound layer optimal cost {c_top:.2f} vs 512 layer {c512:.2f} -> "
              f"{'lower (greed continues to pay, upper bound further out)' if c_top < c512 else 'not lower (greed saturated near 512)'}")
    else:
        print(f"  G2 greedy payoff: upper-bound layer {'no feasible config' if b_top is None else 'comparison missing data'} -- UNDECIDABLE")
    if n_units_top == 0:
        print(f"  G3 signal behavior: winner layer K_t={pick_kt}; upper-bound {kt_top} layer **no data**"
              f" (not collected) -- UNDECIDABLE ('not collected' is not 'blocked')")
    else:
        print(f"  G3 signal behavior: winner layer K_t={pick_kt} vs upper bound {kt_top} -> "
              f"{'took the largest K_t (greed continues)' if pick_kt == kt_top else 'did not take the largest K_t (blocked)'}")
    # ---- R1-R3: K_t=2048 exclusion effect (docstring (15)) ----
    prev_kt = K_TEACHERS[-2]
    prev_layer = [c for c in layers[-2][1] if c in feas]
    print(f"  R1 exclusion effect: upper-bound layer K_t={kt_top} feasible {len(f_top)}/{n_per}"
          f" ({prev_kt} layer {len(prev_layer)}/{n_per}) -> "
          + ("data not collected" if n_units_top == 0 else
             ("gate begins to **exclude** configs (natural upper bound holds)" if len(f_top) < n_per
              else "still not excluded (greedy upper bound must rely on artificial limits)")))
    if f_top and prev_layer:
        e_top = statistics.median([feas[c]["stopped"] for c in f_top])
        e_prev = statistics.median([feas[c]["stopped"] for c in prev_layer])
        if e_top > e_prev:
            r2 = "rebound (gate cost now exceeds the big-codebook speedup -> cost turning point appears)"
        elif e_top == e_prev:
            r2 = "flat (speedup and gate cost cancel)"
        else:
            r2 = "still decreasing (speedup still dominates)"
        print(f"  R2 cost direction: upper-bound layer median epochs-to-gate={e_top} vs {prev_kt} layer={e_prev} -> {r2}")
    else:
        print(f"  R2 cost direction: upper-bound layer has no feasible config or comparison missing data -- UNDECIDABLE")
    vq = VQ_D * kt_top
    cv16 = width_flops_factor(WIDTHS[0])
    print(f"  R3 cost-term weight: VQ term {vq} vs conv(w16) {cv16} = "
          f"+{100.0 * vq / cv16:.0f}% (first four layers <7%); "
          f"gold is now {cfg_tag(gold)} (layer K_t={CONFIGS[gold]['kt']})"
          f" -> {'gold lies in the upper-bound layer' if CONFIGS[gold]['kt'] == kt_top else 'gold not in the upper-bound layer'}")
    print(f"  Q4 conclusion stability: winner K_t={pick_kt} vs v1 anchor layer {K_T} -> "
          f"{'still in the anchor layer (expansion did not change the conclusion)' if pick_kt == K_T else 'left the anchor layer (K_t is a genuine improvement axis, recorded as-is)'}")

    # ---- v3.0 second-task generalization criteria (docstring (16) T1-T4) ----
    print("\n--- v3.0 generalization criteria (docstring (16); task=%s) ---" % TASK)
    collected = set(COLLECT_KT) if COLLECT_KT else set(K_TEACHERS)
    # T1: does the v1.2 normalized signal still give regret=0 on the new task
    print(f"  T1 signal generalization: v1.2(normalized) final pick={cfg_tag(w_tn)} regret={r_tn} "
          f"gold={cfg_tag(gold)} -> "
          f"{'SUPPORT (regret=0, signal isomorphic to objective and task-independent)' if (r_tn == 0 and w_tn == gold) else 'NO SUPPORT (recorded as-is)'}")
    # T2: does the intra-layer optimal normalized cost first decrease then increase with K_t (turning point)
    seq = [(kt, cost_ep[layer_best[kt]]) for kt in K_TEACHERS
           if kt in collected and layer_best.get(kt) is not None]
    # a layer with data but no feasible config = hard gate exclusion (different from "not collected", must be reported separately)
    blocked = [kt for kt in K_TEACHERS if kt in collected
               and layer_best.get(kt) is None
               and layer_units(dict(kt_layers())[kt], seeds, ckpt_root) > 0]
    btxt = f"; gate hard-exclusion layers {blocked} (have data but 0 feasible)" if blocked else ""
    stxt = " ".join(f"{kt}:{v:.1f}" for kt, v in seq)
    if len(seq) < 3:
        t2 = f"UNDECIDABLE (fewer than 3 collected layers: {stxt})"
    else:
        costs = [v for _, v in seq]
        dips = [i for i in range(1, len(costs) - 1)
                if costs[i] < costs[i - 1] and costs[i] < costs[i + 1]]
        upper_missing = [kt for kt in K_TEACHERS if kt > seq[-1][0] and kt not in collected]
        if dips:
            t2 = f"SUPPORT (turning point at K_t={seq[dips[0]][0]}, intra-layer optimal cost first decreases then increases)"
        elif blocked and blocked[0] > seq[-1][0]:
            t2 = (f"HARD TRUNCATION (not a cost turning point): highest feasible layer is K_t={seq[-1][0]}"
                  f" (cost {seq[-1][1]:.1f}), the layers above {blocked} are excluded wholesale by the G0 gate"
                  f" -- greed is capped by the gate rather than by cost (mechanism differs from v2's 'soft upper bound', recorded as-is)")
        elif all(costs[i] >= costs[i + 1] for i in range(len(costs) - 1)):
            t2 = (f"NOT REPRODUCED (monotone decrease within collected layers: {stxt})"
                  + (f"; higher layers {upper_missing} not collected, turning-point location undecidable -- upper bound UNDECIDABLE"
                     if upper_missing else "; no higher layer to test, turning point outside the space"))
        else:
            t2 = f"NO SUPPORT ({stxt} not first-decrease-then-increase)"
    print(f"  T2 greedy self-limiting reproduction: intra-layer optimal normalized cost [{stxt}] -> {t2}{btxt}")
    # T3: K_t=256 layer feasibility + codebook x metric interaction
    l0 = [c for c in layers[0][1] if c in feas]
    n0 = layer_units(layers[0][1], seeds, ckpt_root)
    inter = {kt: metric_interaction(idxs, feas, seeds, ckpt_root)
             for kt, idxs in kt_layers()}
    perfect = [kt for kt, d in inter.items() if d and d["perfect"]]
    mixed = [kt for kt, d in inter.items()
             if d and not d["perfect"] and 0 < d["cos"][0] < d["cos"][1]
             and 0 < d["euc"][0] < d["euc"][1]]
    itxt = (", ".join(f"K_t={kt} perfect separation (cosine {d['cos'][0]}/{d['cos'][1]}, "
                      f"euclid {d['euc'][0]}/{d['euc'][1]})"
                      for kt, d in inter.items() if d and d["perfect"])
            or "no perfect-separation layer")
    nodata = [kt for kt, d in inter.items() if d is None]
    print(f"  T3 G0 boundary reproduction: K_t=256 layer (anchor) feasible {len(l0)}/"
          f"{len(layers[0][1]) if n0 else 0}"
          f"{'  [no data]' if n0 == 0 else ''}; codebook x metric interaction -> {itxt}"
          + (f"; uncollected layers {nodata} UNDECIDABLE" if nodata else "")
          + f"; MIXED layers {mixed if mixed else 'none'}")
    # T4: early-window L_func separability (the pre-registered form = width monotonicity, already judged 'not passing' in calibration;
    #     the operationalization supplement = per-seed rank correlation, honestly annotated as a post-hoc operationalization)
    rows_l = []
    for s in seeds:
        xs, ys = [], []
        for c in POOL:
            if c not in feas:
                continue
            r = load_unit(c, s, RUNGS[-1], ckpt_root)
            if r is None or r["t_fit"] is None:
                continue
            xs.append(l_func_at10(r))
            ys.append(cost_ep[c])
        if len(xs) >= 4:
            rows_l.append(spearman(xs, ys))
    if not rows_l:
        print("  T4 L_func early-window separability: UNDECIDABLE (insufficient feasible units)")
    elif all(r <= -0.5 for r in rows_l):
        print(f"  T4 L_func early-window separability: SUPPORT (per-seed Spearman(L_func@10, normalized cost)="
              f"{['%.2f' % r for r in rows_l]} all <= -0.5: high L_func = low cost, direction usable for pruning)")
    elif all(abs(r) < 0.3 for r in rows_l):
        print(f"  T4 L_func early-window separability: NO SUPPORT (rank correlations {['%.2f' % r for r in rows_l]} "
              f"all <0.3: essentially unrelated to cost, consistent with calibration width monotonicity failing)")
    else:
        print(f"  T4 L_func early-window separability: MIXED ({['%.2f' % r for r in rows_l]}) -- recorded as-is")

    _append_jsonl(jsonl, dict(
        mode="main_summary", n_seeds=len(seeds), gold_cfg=gold, gold_t_fit=gold_t,
        gold_stopped=gold_ep, gold_cfg_wall=gold_w, wall_ep_monotone=mono,
        gold_conv_only=gold_conv, holdout=ho, holdout_norm=ho_n,
        cfg_timenorm=w_tn, rung2_time=r2_t, rung2_timenorm=r2_n,
        rs_median_regret=med_rs, rs_infeasible_rate=inf_rate, n_draw=n_draw,
        rows=rows,
        sh8_trace=[dict(ep=tr["ep"], keep=tr["keep"]) for tr in sh_traces["sh8"]],
        ts=time.strftime("%Y-%m-%d %H:%M:%S")))


# ---------------------------------------------------------------------------
# --ent-ratio: v2.2 systematic test (docstring (14))
# ---------------------------------------------------------------------------

ACC_GATE = 0.95        # consistent with acc_min of mnist_demo.g0_gate
GATE_COEF = 0.8        # codebook gate coefficient (h_z >= GATE_COEF * ln K_t)


def unit_acc_ep_h(rec, kt):
    """The epoch where the unit's acc first reaches >= ACC_GATE and its H(Z_t) (None plus last-epoch H if never).

    The codebook gate is checked exactly **at the moment** acc passes the gate, so only that epoch's H is "the number the gate
    actually compares against". Returns (ep_acc or None, h, ratio = h / ln K_t).
    """
    acc, hz = rec["curve"]["acc"], rec["curve"]["h_z"]
    ep = next((i + 1 for i, a in enumerate(acc) if a >= ACC_GATE), None)
    h = hz[(ep - 1) if ep is not None else len(hz) - 1]
    return ep, h, h / math.log(kt)


def run_ent_ratio(seeds, ckpt_root, jsonl):
    print("=" * 68)
    print(f"v2.2 H(Z_t)/ln K_t systematic test (scheme A; zero training)  seeds={seeds}  "
          f"[smoke={QUICK}]")
    print(f"Reading: the H(Z_t) of the epoch where acc first reaches >= {ACC_GATE}; codebook gate coefficient = {GATE_COEF}")
    print("=" * 68)
    passed, unpassed = {}, {}
    for kt, idxs in kt_layers():
        passed[kt], unpassed[kt] = [], []
        for c in idxs:
            for s in seeds:
                rec = load_unit(c, s, RUNGS[-1], ckpt_root)
                if rec is None:
                    continue
                ep, h, r = unit_acc_ep_h(rec, kt)
                (passed if ep is not None else unpassed)[kt].append(
                    (c, s, ep, h, r))
    lnk = {kt: math.log(kt) for kt in K_TEACHERS}
    print("\n--- per layer (acc-gated units only; ungated units listed separately at the end) ---")
    print(f"  {'K_t':>5}{'n':>4}{'ln K_t':>9}{'gate':>8}{'median H':>8}"
          f"{'median ratio':>10}   H range")
    med = {}
    for kt in K_TEACHERS:
        u = passed[kt]
        if not u:
            print(f"  {kt:>5}{0:>4}{lnk[kt]:>9.3f}{GATE_COEF * lnk[kt]:>8.3f}"
                  f"       -         -   no acc-gated units")
            continue
        hs = [x[3] for x in u]
        med[kt] = statistics.median([x[4] for x in u])
        print(f"  {kt:>5}{len(u):>4}{lnk[kt]:>9.3f}{GATE_COEF * lnk[kt]:>8.3f}"
              f"{statistics.median(hs):>8.3f}{med[kt]:>10.4f}"
              f"   [{min(hs):.2f}, {max(hs):.2f}]")
    allr = [x[4] for kt in K_TEACHERS for x in passed[kt]]
    if not allr:
        print("\nNo acc-gated units -- UNDECIDABLE.")
        return
    if len(med) >= 2:
        spread = max(med.values()) - min(med.values())
        print(f"\nH1 tracking hypothesis: inter-layer median-ratio spread = {spread:.4f} -> "
              f"{'SUPPORT (approximately constant: H rises in sync with the gate, the gate never bites at any K_t)' if spread < 0.05 else 'NO SUPPORT (must be described per layer, not characterizable by a single coefficient)'}")
    else:
        spread = None
        print("\nH1 tracking hypothesis: insufficient layers -- UNDECIDABLE")
    print(f"H2 bitable coefficient: max ratio over all gated units = {max(allr):.4f} (current coefficient "
          f"{GATE_COEF}) -> coefficient must exceed {max(allr):.3f} before the gate starts blocking")
    umin = min((x for kt in K_TEACHERS for x in passed[kt]), key=lambda x: x[2])
    print(f"     fastest gated unit: {cfg_tag(umin[0])} s{umin[1]}  ep{umin[2]}  "
          f"H={umin[3]:.3f}  ratio={umin[4]:.4f}  "
          f"(gate of that layer {GATE_COEF * math.log(CONFIGS[umin[0]]['kt']):.3f})")
    n_un = sum(len(unpassed[kt]) for kt in K_TEACHERS)
    if n_un:
        unr = [x[4] for kt in K_TEACHERS for x in unpassed[kt]]
        print(f"\nUnits not passing the acc gate (listed separately, not mixed into H1/H2): n={n_un}  "
              f"median last-epoch ratio={statistics.median(unr):.4f}  "
              f"range [{min(unr):.4f}, {max(unr):.4f}]")
    _append_jsonl(jsonl, dict(
        mode="ent_ratio", n_seeds=len(seeds), spread=spread,
        max_ratio=max(allr), n_unpassed=n_un,
        per_layer={str(kt): dict(
            n=len(passed[kt]), gate=GATE_COEF * lnk[kt],
            med_h=(statistics.median([x[3] for x in passed[kt]])
                   if passed[kt] else None),
            med_ratio=med.get(kt)) for kt in K_TEACHERS},
        ts=time.strftime("%Y-%m-%d %H:%M:%S")))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    global QUICK
    ap = argparse.ArgumentParser(description="optimizer v1: SH + three-layer reward")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--calibrate", action="store_true", help="axis B calibration")
    g.add_argument("--search", action="store_true", help="SH search")
    g.add_argument("--report", action="store_true", help="summary report")
    g.add_argument("--main", action="store_true",
                   help="main experiment: exhaustive gold + offline replay comparison (optimizer effectiveness)")
    g.add_argument("--ent-ratio", action="store_true",
                   help="H(Z_t)/ln K_t systematic test (docstring (14), zero training)")
    ap.add_argument("--rho", action="store_true",
                    help="with --report: also run the winner rho audit")
    ap.add_argument("--signal", choices=["acc", "time", "time_norm"],
                    default="time_norm",
                    help="search pruning signal: time_norm=v1.2 normalized (default), time=v1.1, acc=v1.0")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds; --main defaults to 0-7, others default to 0,1,2")
    ap.add_argument("--resume", action="store_true", help="resume training from checkpoints")
    ap.add_argument("--task", choices=TASKS, default="grid64",
                    help="task: grid64=existing archive (position encoding), shape64=second task (shape encoding)")
    ap.add_argument("--layers", default=None,
                    help="comma-separated K_t layers, limit Phase 1 collection (default: all layers)")
    ap.add_argument("--replay-only", action="store_true",
                    help="with --main: skip Phase 1 collection, offline replay verdicts only")
    ap.add_argument("--quick", action="store_true", help="smoke (writes to an isolated *_quick directory)")
    args = ap.parse_args()

    QUICK = args.quick
    global TASK, COLLECT_KT, RUNGS
    TASK = args.task
    if args.layers:
        COLLECT_KT = {int(x) for x in args.layers.split(",") if x != ""}
    jsonl, log, ckpt_root = _paths()
    os.makedirs(ckpt_root, exist_ok=True)
    _redirect_log(log)
    spec = args.seeds if args.seeds is not None else ("0,1,2,3,4,5,6,7" if args.main
                                                     else "0,1,2")
    seeds = [int(x) for x in spec.split(",") if x != ""]
    if QUICK:
        RUNGS = [3]        # smoke single rung (no decision, zero sharing with official data)
    else:
        # task budget (revision of the docstring (16) calibration clause, before any shape64 decision data):
        # the shape64 calibration probes all pass G0, but the reference config w16 uses up 26/40 = 65% of the budget
        # (the same config in grid64 6/40 = 15%) -> under 40ep, "infeasible" would be mixed with insufficient budget, making the T3
        # boundary inexplicable. So shape64 uses a full budget of 80ep, rung = [20, 40, 80];
        # grid64 keeps 40ep -> rung = [10, 20, 40] bit-for-bit unchanged.
        ep_full = 80 if TASK == "shape64" else 40
        RUNGS = [ep_full // 4, ep_full // 2, ep_full]

    if args.calibrate:
        run_calibrate(seeds, args.resume, jsonl, ckpt_root)
    elif args.search:
        run_search(seeds, args.resume, jsonl, ckpt_root, signal=args.signal)
    elif args.main:
        run_main(seeds, args.resume, jsonl, ckpt_root,
                 replay_only=args.replay_only)
    elif args.ent_ratio:
        run_ent_ratio(seeds, ckpt_root, jsonl)
    else:
        run_report(seeds, args.rho, args.resume, jsonl, ckpt_root)


if __name__ == "__main__":
    main()
