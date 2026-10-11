"""The criteria and the preflight checklist (preflight.py) -- the project's headline rules.

excluded_set / coverage / false_exclusions were promoted into the package on 2026-09-30 precisely because
they ARE the product rule (`log K < R_IB => infeasible`), so they are guarded here.
"""

import math

from support import approx

import capacity_lens as C


# ---------------------------------------------------------------------------
# the headline rule and its two validation metrics
# ---------------------------------------------------------------------------

def test_excluded_set_uses_log_k_strictly_below_rib():
    assert C.excluded_set(2.0, [2, 4, 8, 16]) == [2, 4]
    assert C.excluded_set(math.log(8), [2, 4, 8]) == [2, 4]        # equality is NOT excluded
    assert C.excluded_set(0.0, [2, 4]) == []


def test_excluded_set_is_direction_safe_on_the_grid():
    # a K above the requirement is never excluded, no matter how the grid is ordered
    for K in (128, 512, 4096):
        assert K not in C.excluded_set(1.0, [2, 8, 128, 512, 4096])


def test_coverage_counts_only_measured_failures():
    assert C.coverage([2, 4], [2, 4, 8]) == approx(2 / 3)
    assert C.coverage([2, 4], [2, 4]) == approx(1.0)
    assert C.coverage([], [2]) == approx(0.0)
    assert C.coverage([2], []) == approx(1.0)               # nothing to cover -> full by convention


def test_false_exclusions_are_the_excluded_sizes_that_work():
    assert C.false_exclusions([2, 4, 8], [4, 8]) == [4, 8]
    assert C.false_exclusions([2, 4], [8, 16]) == []


# ---------------------------------------------------------------------------
# measured wall and looseness
# ---------------------------------------------------------------------------

def test_measured_wall_is_the_smallest_k_retaining_the_reference():
    grid = [2, 4, 8, 16, 32, 64]
    accs = {2: 0.20, 4: 0.40, 8: 0.90, 16: 0.95, 32: 0.97, 64: 0.98}
    k_wall, rel = C.measured_wall(grid, accs, 0.98, rel_tol=0.10)
    assert k_wall == 8                                             # 0.90 >= 0.98 * (1 - 0.10)
    assert rel[8] == approx((0.98 - 0.90) / 0.98)


def test_measured_wall_is_relative_so_it_survives_a_weak_reference():
    grid = [2, 4, 8]
    accs = {2: 0.05, 4: 0.115, 8: 0.125}                           # 0.115 is within 10% of the reference
    k_wall, _ = C.measured_wall(grid, accs, 0.125, rel_tol=0.10)
    assert k_wall == 4                                             # an absolute 0.25 gap would be vacuous


def test_measured_wall_returns_none_when_no_grid_point_matches():
    k_wall, _ = C.measured_wall([2, 4], {2: 0.1, 4: 0.2}, 0.9)
    assert k_wall is None


def test_wall_band_reports_the_tolerance_sensitivity():
    """The wall is a threshold crossing, so its value depends on the tolerance; the band makes that explicit.

    Motivation (measured, SCALE-AXES A2): on a flat gap curve a 2-point retention difference moved the located
    wall from 2048 to 512, so a single number can be off by a factor of four.
    """
    grid = [8, 16, 32, 64]
    steep = {8: 0.20, 16: 0.85, 32: 0.92, 64: 0.95}
    flat = {8: 0.42, 16: 0.43, 32: 0.445, 64: 0.48}
    b_steep = C.k_wall_band(grid, steep, 0.95)
    b_flat = C.k_wall_band(grid, flat, 0.50)
    assert set(b_steep) == {0.05, 0.10, 0.15}                      # the default tolerances, in order
    assert max(v for v in b_steep.values() if v) / min(v for v in b_steep.values() if v) < 4
    assert max(v for v in b_flat.values() if v) / min(v for v in b_flat.values() if v) >= 4
    # the band is monotone in the tolerance: a stricter retention bar never needs fewer codes
    assert b_flat[0.05] >= b_flat[0.10] >= b_flat[0.15]


def test_preflight_report_exposes_the_band_and_warns_when_it_is_wide():
    grid = [8, 16, 32, 64]
    flat = {8: 0.42, 16: 0.43, 32: 0.445, 64: 0.48}
    rep = C.preflight_report(0.50, 5.0, 256, grid, flat, acc_min=0.4, r_ib=1.36)
    assert "wall_band" in rep and rep["items"]["wall_band_wide"] is True
    assert "TOLERANCE-SENSITIVE" in rep["verdict"]
    steep = {8: 0.20, 16: 0.85, 32: 0.92, 64: 0.95}
    rep2 = C.preflight_report(0.95, 5.0, 256, grid, steep, acc_min=0.4, r_ib=1.36)
    assert not rep2["items"].get("wall_band_wide")
    assert "TOLERANCE-SENSITIVE" not in rep2["verdict"]


def test_looseness_is_log_ratio_and_none_when_undefined():
    assert C.looseness(2.0, 8) == approx(math.log(8) / 2.0)
    assert C.looseness(0.0, 8) is None
    assert C.looseness(2.0, None) is None
    assert C.looseness(2.0, 1) is None                             # K < 2 is not a codebook


def test_looseness_is_one_when_the_bound_coincides_with_the_wall():
    assert C.looseness(math.log(64), 64) == approx(1.0)


# ---------------------------------------------------------------------------
# the preflight checklist
# ---------------------------------------------------------------------------

def _flattening_accs(smooth=True):
    grid = [2, 4, 8, 16, 32, 64]
    accs = {2: 0.3, 4: 0.9, 8: 0.95, 16: 0.96, 32: 0.965, 64: 0.97}
    if smooth:
        accs = {2: 0.2, 4: 0.55, 8: 0.85, 16: 0.94, 32: 0.965, 64: 0.97}
    return grid, accs


def test_g0_gate_requires_both_accuracy_and_codebook_usage():
    ok, ok_acc, ok_cb = C.g0_gate(0.97, math.log(200), 256)
    assert ok and ok_acc and ok_cb
    bad, ok_acc2, _ = C.g0_gate(0.80, math.log(200), 256)
    assert not bad and not ok_acc2
    bad2, _, ok_cb2 = C.g0_gate(0.97, 0.1, 256)
    assert not bad2 and not ok_cb2


def test_plateau_and_gap_k_star():
    grid, accs = _flattening_accs(smooth=True)
    assert C.distill_plateau_ok(grid, accs)
    k_star = C.distill_gap_k_star(0.97, grid, accs)
    assert k_star in grid
    assert k_star >= 16                                            # the gap closes late, not at the bottom


def test_plateau_is_refused_when_the_curve_is_still_rising():
    grid = [2, 4, 8, 16]
    accs = {2: 0.2, 4: 0.5, 8: 0.65, 16: 0.95}                     # top gap still shrinking fast
    assert not C.distill_plateau_ok(grid, accs)
    # the refusal lives in the caller: without a plateau the report must refuse to hand out K*
    rep = C.preflight_report(0.95, math.log(240), 256, grid, accs, acc_min=0.9)
    assert rep["k_star"] is None and rep["degraded"]
    assert "plateau not established" in rep["verdict"]


def test_preflight_report_reports_the_practical_wall_and_looseness():
    grid, accs = _flattening_accs(smooth=True)
    rep = C.preflight_report(0.97, math.log(240), 256, grid, accs, acc_min=0.95, r_ib=1.36)
    assert rep["k_wall"] == 16                                     # 0.94 vs the 0.97 reference: within 10%
    assert rep["looseness"] == approx(math.log(16) / 1.36)
    assert rep["items"]["practical_wall"] == 16
    assert "bound_looseness" in rep["items"]
    assert "practical wall" in rep["verdict"]


def test_preflight_report_restricts_when_the_gate_fails():
    grid, accs = _flattening_accs(smooth=True)
    rep = C.preflight_report(0.90, math.log(240), 256, grid, accs, acc_min=0.95, r_ib=1.36)
    assert rep["degraded"]
    assert rep["verdict"].startswith("RESTRICTED")


def test_preflight_report_diffuse_wall_only_promises_exclusion():
    grid, accs = _flattening_accs(smooth=True)
    rep = C.preflight_report(0.97, math.log(240), 256, grid, accs, acc_min=0.95,
                             wall_probe={"wall": "diffuse"}, r_ib=1.36)
    assert "NOT K* location" in rep["verdict"]


def test_ratio_verdict_vocabulary_and_direction():
    good = [(0, 2.0, 2.1), (1, 3.0, 3.2), (2, 1.5, 1.6)]
    bad = [(0, 1.0, 50.0), (1, 1.0, 60.0), (2, 1.0, 70.0)]
    v_good = C.ratio_verdict(good, "unit")[0]        # returns (verdict, detail)
    v_bad = C.ratio_verdict(bad, "unit")[0]
    assert v_good.split(":")[0] in {"PASS", "FAIL", "PARTIAL", "UNDECIDABLE"}
    assert not v_bad.startswith("PASS")
