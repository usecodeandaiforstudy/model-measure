"""Judgment preconditions (pure functions): the checklist a distillation/quantization verdict must pass before
its numbers may be trusted.

These are the CANONICAL implementations, moved 2026-09-30 from experiments/mnist_demo.py into the package
(the E-001 methodology lesson, productized: never report a verdict without self-checking the intermediate
quantities it depends on -- the router-purity analogue for capacity judgments).

- g0_gate: the teacher really encoded the task AND its codebook has not collapsed;
- distill_plateau_ok / distill_gap_k_star: the gap-curve reference has converged before locating K*;
- ratio_verdict: the multi-seed consistency arbitration (octave window);
- preflight_report: the five-item checklist as one call (items -> verdict -> degraded?).
"""

import math


def g0_gate(acc, h_z, k_teacher, acc_min=0.95, usage_min=0.8):
    """G0 double gate: acc gate + codebook utilization gate (pure function, independently unit-testable).

    acc gate: teacher acc >= acc_min (encoding validity).
    Codebook gate (added 2026-09-22): H(Z_t) >= usage_min * ln(k_teacher).

    Motivation (CP3 re-review of the C3 v3 raw logs): the old single gate let through a collapsed teacher
    with acc=0.961 (right at the gate) but H(Z_t)=4.18=ln64——only ~65 of the 256 codes were used, the teacher
    itself was sitting on the wall, and using it to measure "whether there is a cliff below the wall" fails by
    definition. The measured utilization distribution is well separated: pathological (C3 v3 / CP1 arm 2)
    H(Z_t)/ln256 ~= 0.75, healthy (CP3 calibrated version) ~= 0.93+, with the threshold 0.8 in between.
    Retroactivity: only constrains future runs; all archived teachers (12 in CP3, H(Z_t) 5.18-5.39) pass this
    gate, so the archived conclusions are unchanged.

    Gate calibration discipline (CP-H, 2026-09-29): acc_min must be chosen against the QUANTIZED K_min
    regimes, not against a noise band -- on CIFAR-10 acc>=0.715 gives K_min>=4 (non-degenerate wall {2,3}),
    acc>=~0.807 gives K_min>=5; a gate inside the run-to-run noise band (+-0.03 measured) tests noise, not
    strength. Never re-roll a failing seed to pass the gate (selection on noise).

    Returns a triple (overall verdict, acc gate, codebook gate).
    """
    ok_acc = acc >= acc_min
    ok_cb = h_z >= usage_min * math.log(k_teacher)
    return (ok_acc and ok_cb), ok_acc, ok_cb


def distill_gap_k_star(acc_t, k_list, accs, tol=0.02):
    """Gives the actual minimum usable student K* from the **gap curve** (pure function, independently unit-testable).

    Gap g(K) = acc_t - acc_s(K). Reference gap g_ref = g(K_max) (the student with the largest K in the grid):
    it is the least capacity-limited, so its gap is the **irreducible cost** of distillation itself
    (the KD objective + finite training budget).
    K* = the smallest K satisfying g(K) <= g_ref + tol; None if the set is empty.

    Why not continue using "acc_s >= acc_t - 0.03" (absolute threshold): the student's **systematic gap**
    relative to the teacher is of the same order as 0.03, so "whether any student meets the bar" is decided by
    how well the teacher happened to train this time——measured: teacher rising from 0.864 to 0.884 (only 0.020
    stronger) flipped K* from 132 into "no one in the whole grid meets the bar". After switching to using the
    gap of the largest-K student **within the same run** as the reference, the teacher's strength is canceled
    out: g(K) - g_ref is independent of the teacher's absolute level, and the criterion is no longer sensitive
    to the teacher's strength.
    """
    ks = sorted(k_list)
    if not ks:
        return None
    g_ref = acc_t - accs[ks[-1]]
    usable = [K for K in ks if (acc_t - accs[K]) <= g_ref + tol]
    return min(usable) if usable else None


def excluded_set(rib, k_list):
    """The headline rule, as a function: E = {K in k_list : log K < R_IB} -- the sizes the certified
    requirement bound rules out.

    Direction-safe: R_IB comes from an ACHIEVED error, so it is a lower bound on the requirement; the rule
    can therefore only under-state what is impossible, never over-state it (a K it excludes is infeasible;
    a K it keeps is merely not-yet-ruled-out -- it is NOT a claim that K suffices). Pure function.
    """
    return sorted(K for K in k_list if math.log(K) < rib)


def coverage(excl, fails):
    """Fraction of MEASURED failures that the certified exclusion already rules out (the exclusion's
    usefulness; 1.0 when there is nothing to cover). Pure function."""
    if not fails:
        return 1.0
    return len([K for K in excl if K in fails]) / len(fails)


def false_exclusions(excl, succeeds):
    """The excluded sizes that were MEASURED to work -- the error the rule must never make (a false
    exclusion kills a usable configuration). Pure function."""
    return sorted(K for K in excl if K in succeeds)


def measured_wall(k_list, accs, acc_ref, rel_tol=0.10):
    """The PRACTICAL wall: the smallest K whose student retains (1 - rel_tol) of the reference accuracy.

    Why this exists (MOE and WITNESS, 2026-09-30): the certified Fano bound is SAFE (a lower bound on the
    task's information requirement) but often much LOOSER than the capacity that actually governs a
    deployment decision. Two measurements forced this function into existence:
      - MOE (route64): the two-tier flow bound predicted the routed network's curve only once anchored at
        the task's MEASURED dense wall, not at the Fano bound (which underestimated the wall by 4x);
      - WITNESS (clean/merged grid64 + CIFAR-10): on the real task the strongest witness excluded {2,3}
        while measured success started at K=128 -- sound, but far from the wall.
    Hence a report should carry BOTH numbers: the certified exclusion (log K < R_IB) and the practical
    wall (K < K_wall).

    The criterion is the RELATIVE gap, which makes it scale-free: an absolute gap of 0.25 is unreachable
    when the reference itself scores 0.125 (WITNESS task B) and would leave the measurement vacuous.
    Different from distill_gap_k_star, which uses the within-run irreducible KD gap as its reference;
    this one answers "where does the student stop being distinguishable from the reference at all".
    Returns (k_wall or None, rel_gaps dict); k_wall is None when no grid point matches the reference.
    """
    ks = sorted(k_list)
    denom = max(float(acc_ref), 1e-12)
    rel = {K: (float(acc_ref) - float(accs[K])) / denom for K in ks}
    ok = [K for K in ks if rel[K] <= rel_tol]
    return (min(ok) if ok else None), rel


def k_wall_band(k_list, accs, acc_ref, rel_tols=(0.05, 0.10, 0.15)):
    """The practical wall at several retention tolerances, instead of a single number.

    Why this exists (measured, SCALE-AXES A2): the located wall is the smallest K that crosses a retention
    threshold, so on a FLAT part of the gap curve a couple of percentage points of retention move it by a factor
    of 4 (measured: 2048 at rel_tol 0.10 vs 512 once the student retained 2 points more). A single number
    therefore hides how much of the "wall" is the criterion rather than the capacity. The band makes the
    sensitivity explicit: a narrow band means the wall is steep and meaningful; a wide band means it should be
    quoted as an interval, or not at all.

    Args:
        k_list: the K grid (ascending)
        accs: {K: accuracy} of the students trained at those K
        acc_ref: the reference accuracy
        rel_tols: the retention tolerances to report (default (0.05, 0.10, 0.15))
    Returns:
        {rel_tol: k_wall or None} in the order given
    """
    return {tol: measured_wall(k_list, accs, acc_ref, rel_tol=tol)[0] for tol in rel_tols}


def looseness(rib, k_wall):
    """How much looser the certified bound is than the practical wall, in nats: ln(k_wall) / rib.

    1.0 = the certified bound coincides with the practical wall; >1 = the certified bound sits that many
    times lower than the capacity that actually matters (so exclusions stated from it are sound but
    conservative). None when either side is undefined (no wall located, or rib <= 0)."""
    if k_wall is None or k_wall < 2:
        return None
    rib = float(rib)
    if rib <= 0.0:
        return None
    return math.log(k_wall) / rib


def distill_plateau_ok(k_list, accs, tol=0.02):
    """Whether the reference gap has already plateaued (pure function, independently unit-testable).

    Plateau condition: the difference between the gaps of the top two levels of the grid is <= tol, i.e.
    g(K_second_largest) - g(K_max) <= tol, equivalent to acc_s(K_max) - acc_s(K_second_largest) <= tol
    (the top accuracy no longer rises appreciably).

    Why this pre-check is mandatory: if the top gap is still decreasing appreciably, the "irreducible cost"
    has not yet been reached and g_ref is too small, so only the very top K falls within g_ref + tol and K* is
    pushed to the upper end of the grid, which instead **necessarily falls within one octave of K_min**——
    a trap that produces false passes. When the plateau is not established, K* should be refused (recorded as
    unlocated) rather than given anyway.

    (Known edge, CP-H 2026-09-30: on real tasks the top two gaps may hover just above tol while the KD
    student is still converging -- 0.025 vs 0.02 measured; the honest behavior is refusal, plus the
    wall-geometry probe to tell the user that K* location may not apply at all.)
    """
    ks = sorted(k_list)
    if len(ks) < 2:
        return True
    return (accs[ks[-1]] - accs[ks[-2]]) <= tol


def ratio_verdict(records, label, octave=2.0, min_pass_frac=2 / 3):
    """Multi-seed consistency verdict (pure function, independently unit-testable; shared by feature A and feature B).

    records: [(seed, predicted_value, measured_value or None), ...] (measured value None = that seed did not locate a cliff).

    A seed passes when its ratio measured/predicted falls in [1/octave, octave]. A **pass** verdict requires all of:
      1. the median ratio falls within the window;
      2. the in-window fraction >= min_pass_frac;
      3. **no seed is "failed to locate a cliff"**.
    Item 3 prevents failures like "in some seed the student simply never met the bar / the quantizer was never
    pinched" from being silently discarded——silent exclusion systematically biases toward passing
    (treating unfavorable evidence as missing data).

    Returns (verdict, detail_lines).
    """
    detail, ratios, seeds_out, seeds_none = [], [], [], []
    for seed, pred, obs in records:
        if obs is None:
            seeds_none.append(seed)
            detail.append(f"  seed {seed}: predicted {pred:.2f} / measured: no cliff located")
            continue
        r = obs / pred
        ok = (1.0 / octave) <= r <= octave
        ratios.append(r)
        if not ok:
            seeds_out.append(seed)
        detail.append(f"  seed {seed}: predicted {pred:.2f} / measured {obs:.2f} -> ratio {r:.2f} "
                      f"{'PASS' if ok else 'FAIL'}")
    n_total = len(records)
    if not ratios:
        detail.append("  no seed located a cliff")
        return "UNDECIDABLE: no seed located a cliff (protocol or configuration problem)", detail
    ratios.sort()
    med = ratios[len(ratios) // 2]
    frac = (len(ratios) - len(seeds_out)) / len(ratios)
    detail.append(f"  median ratio = {med:.2f} ({label}); valid seeds {len(ratios)}/{n_total};"
                  f" in-window fraction = {frac:.0%} (outside: {seeds_out if seeds_out else 'none'};"
                  f" no cliff located: {seeds_none if seeds_none else 'none'})")
    in_window = (1.0 / octave) <= med <= octave
    if not in_window:
        direction = ("measured >> predicted (bound too tight / over-optimistic)"
                     if med > octave else
                     "measured << predicted (bound too loose / over-pessimistic)")
        return f"FAIL: median ratio {med:.2f} outside one octave -- {direction}", detail
    if seeds_none:
        return (f"PARTIAL: median ratio {med:.2f} in window, but {len(seeds_none)}/{n_total} seeds"
                f" (seeds {seeds_none}) failed to locate a cliff -- criterion unstable,"
                f" not enough to claim PASS (do not drop these seeds as missing data)"), detail
    if frac < min_pass_frac:
        return (f"PARTIAL: median ratio {med:.2f} in window, but only {frac:.0%} of seeds passed"
                f" individually (< {min_pass_frac:.0%}) -- spread too large,"
                f" single-run conclusions unreliable"), detail
    return f"PASS: median ratio {med:.2f} within one octave, {frac:.0%} of seeds passed individually", detail


def preflight_report(acc_t, h_z_t, k_teacher, k_list, accs, acc_min=0.95, usage_min=0.8,
                     plateau_tol=0.02, k_star_tol=0.02, wall_probe=None, rel_tol=0.10, r_ib=None):
    """The preflight checklist for a distillation verdict (pure function).

    Five core items (each binary, any failure degrades the verdict instead of silently passing):
      1. teacher_encoded    -- the reference model really learned the task (acc gate);
      2. codebook_healthy   -- its codebook has not collapsed (H(Z_t) gate);
      3. plateau_established-- the gap-curve reference has converged (no false K*);
      4. k_star_located     -- K* could actually be located (None = refused, not silently given);
      5. gate_pass          -- the G0 double gate as a whole.
    A sixth, OPTIONAL item (pass wall_probe=wall_geometry_probe(...)):
      6. wall_geometry      -- task wall class from the calibrated probe (A-003): on a "diffuse" wall the
                               report explicitly states that K* location is NOT promised (only exclusion),
                               even if the other items look green.
    A seventh, OPTIONAL item (always attempted, from the grid itself):
      7. practical_wall     -- measured_wall(k_list, accs, acc_t, rel_tol): the K at which the student stops
                               being distinguishable from the reference. Pass r_ib=<the task's Fano bound>
                               to also get bound_looseness = ln(K_wall)/R_IB, i.e. how much lower the
                               certified exclusion sits than the wall. Rationale (MOE + WITNESS, 2026-09-30):
                               R_IB is sound but can be far looser than the capacity that decides a
                               deployment question, so a report should carry both numbers.
    And its sensitivity band (item wall_band, key `wall_band`, flag `wall_band_wide`): the located wall at
    rel_tol in {0.05, 0.10, 0.15}. Measured motivation (SCALE-AXES A2): on a flat gap curve a 2-point retention
    difference moved the located wall from 2048 to 512, so a wall quoted without its tolerance band can be off
    by a factor of four. When the band moves, the verdict says so explicitly.
    Returns dict(items, k_star, degraded, verdict, k_wall, looseness, wall_band). Any failure must surface to
    the user (the E-001 methodology lesson, capacity-judgment version).
    """
    g0, ok_acc, ok_cb = g0_gate(acc_t, h_z_t, k_teacher, acc_min=acc_min, usage_min=usage_min)
    plateau = distill_plateau_ok(k_list, accs, tol=plateau_tol)
    k_star = distill_gap_k_star(acc_t, k_list, accs, tol=k_star_tol) if plateau else None
    k_wall, _rel = measured_wall(k_list, accs, acc_t, rel_tol=rel_tol)
    band = k_wall_band(k_list, accs, acc_t)
    # "wide" = the located wall swings by a factor of 4 or more across the tolerances. Tightening the
    # tolerance always raises the wall a little, so the flag must not fire on that; the case worth warning
    # about is the one the A2 lesson measured -- where the wall is off by a factor of four.
    located = [v for v in band.values() if v is not None]
    band_wide = (max(located) / min(located) >= 4.0) if len(located) >= 2 else False
    lo = looseness(r_ib, k_wall) if r_ib is not None else None
    items = dict(teacher_encoded=bool(ok_acc), codebook_healthy=bool(ok_cb),
                 plateau_established=bool(plateau), k_star_located=(k_star is not None),
                 gate_pass=bool(g0))
    if k_wall is not None:
        items["practical_wall"] = k_wall
    items["wall_band"] = {f"{tol:.2f}": band[tol] for tol in band}
    if band_wide:
        items["wall_band_wide"] = True
    if lo is not None:
        items["bound_looseness"] = round(lo, 3)
    degraded = not (g0 and plateau)
    wall_kind = wall_probe.get("wall") if wall_probe else None
    if wall_kind is not None:
        items["wall_geometry"] = wall_kind
    wall_note = (f"; practical wall K_wall={k_wall}"
                 + (f" (certified bound sits x{lo:.1f} lower)" if lo is not None else "")
                 ) if k_wall is not None else ""
    if band_wide:
        wall_note += (f"; WALL IS TOLERANCE-SENSITIVE {band} -- quote it as an interval, not a number "
                      f"(see k_wall_band)")
    if not g0:
        verdict = ("RESTRICTED: G0 gate failed (teacher did not encode the task or codebook "
                   "collapsed) -- do not reroll to pass the gate (noise selection)")
    elif wall_kind == "diffuse":
        verdict = (f"RESTRICTED (diffuse wall): this task only promises exclusion (K_min + "
                   f"certified screening), NOT K* location"
                   f"{'; current K*=' + str(k_star) + ' reference only' if k_star is not None else '; plateau not established, K* refused per protocol'}"
                   f"{wall_note}")
    elif not plateau:
        verdict = ("RESTRICTED: plateau not established -- K* refused per protocol; run the "
                   "wall-geometry probe to check whether K* location applies to this task"
                   f"{wall_note}")
    else:
        verdict = f"VERDICT USABLE: K*={k_star}{wall_note}"
    return dict(items=items, k_star=k_star, k_wall=k_wall, looseness=lo, wall_band=band,
                degraded=degraded, verdict=verdict)
