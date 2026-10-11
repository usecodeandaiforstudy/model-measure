"""The pre-registered criterion functions that the experiment modules export.

Rationale: these functions ARE the published rules (threshold tables, verdict aggregation, the scaling
fit), so they get a guarded home here. Each experiment still has its own `--selftest` covering its full
verdict assembly; this file covers the cross-script helpers plus the four verdicts whose row shapes are
stable, and the cached-target KD replica used by the third SCALE-C run.
"""

import math
import os
import sys

from support import approx
import torch
from torch.utils.data import DataLoader, TensorDataset

import capacity_lens as C

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import witness_demo as W
import looseness_demo as L
import kv_demo as K
import family_library_demo as F
import pretrained_witness_demo as P
import scale_c_demo as S
import scale_c_clean_demo as SC
import wall_saturation_demo as WS
import cross_task_demo as CT
import scale_axes_demo as SA
import looseness_reaudit_demo as LR
import budget_dose_demo as BD
import family_b_demo as FB
import budget_mechanism_demo as BM
import wall_remeasure_demo as WR
import family_c_demo as FC


# ---------------------------------------------------------------------------
# WITNESS
# ---------------------------------------------------------------------------

def test_is_monotone():
    assert W.is_monotone([1.0, 1.5, 1.5])
    assert not W.is_monotone([1.5, 1.0])


def test_witness_verdict_vocabulary():
    def per_task(cov_w1, false_any):
        return {"A": dict(seeds=[dict(ribs={"W1": 1.0, "W2": 1.2, "W3": 1.5, "W4": 2.0},
                                       false_excl_any=false_any, cov={"W1": cov_w1, "W4": 1.0})],
                          rib_med={"W1": 1.0, "W2": 1.2, "W3": 1.5, "W4": 2.0},
                          kmin_med={"W4": 8}, labelonly_false=0, w4_false=0)}
    assert W.witness_verdict(per_task(0.2, False))[0].startswith("SUPPORT")
    assert W.witness_verdict(per_task(0.95, False))[0].startswith("SHALLOW")
    assert W.witness_verdict(per_task(0.2, True))[0].startswith("UNSOUND")


# ---------------------------------------------------------------------------
# LOOSENESS
# ---------------------------------------------------------------------------

def test_rank_correlation_and_median():
    assert L.rank_correlation([1, 2, 3], [1, 2, 3]) == approx(1.0)
    assert L.rank_correlation([1, 2, 3], [3, 2, 1]) == approx(-1.0)
    assert L.rank_correlation([1.0, 2.0], [1.0, 2.0]) == 0.0          # needs >= 3 points
    assert L.median([3.0, 1.0, 2.0]) == 2.0


def test_loos_verdict_requires_all_expectations():
    det = dict(source="t", name="n", acc_ref=1.0, eps=0.0, n_classes=64, grid=[2, 64], k_min=64,
               k_wall=64, wall_ratio=1.0, looseness=1.0, rib=4.16, note="", min_rel_gap=0.0)
    mid = dict(det, c=1, acc_ref=0.9, eps=0.10, looseness=2.0)
    hi = dict(det, acc_ref=0.7, eps=0.30, looseness=5.0)
    verdict, _ = L.loos_verdict([det, mid, hi])
    assert verdict.split(":")[0] in {"SUPPORT", "PARTIAL", "MIXED", "MIXED/REFUTED"}


# ---------------------------------------------------------------------------
# KV
# ---------------------------------------------------------------------------

def test_kv_helpers_and_octave():
    assert K.dense_requirement_nats(4) == approx(4 * math.log(8))
    assert K.octave_ok(128, math.log(256))
    assert not K.octave_ok(32, math.log(256))
    assert K.smallest_good({8: 0.5, 16: 0.91, 64: 0.99}) == 16
    assert K.smallest_good({8: 0.5}) is None


def test_kv_verdict_needs_the_per_token_mechanism():
    rows = []
    for L_ in [2, 4, 8, 16]:
        rows.append(dict(arm="kv", L=L_, K=8, acc=0.99))
        rows.append(dict(arm="kv", L=L_, K=4, acc=0.25))
    for L_ in [2, 3, 4]:
        for K_ in [64, 96, 128, 192, 512, 768, 1024, 2048, 4096, 8192]:
            rows.append(dict(arm="dense", L=L_, K=K_, acc=0.95 if K_ >= 8 ** L_ else 0.2))
    for L_ in [2, 4, 8]:
        rows.append(dict(arm="ctrl", L=L_, K=8, acc=0.95))
    assert K.kv_verdict(rows)[0].startswith("SUPPORT")
    broken = [dict(r) for r in rows if not (r["arm"] == "kv" and r["K"] == 4)]
    broken += [dict(arm="kv", L=L_, K=4, acc=0.99) for L_ in [2, 4, 8, 16]]
    assert not K.kv_verdict(broken)[0].startswith("SUPPORT")


# ---------------------------------------------------------------------------
# FAMILY LIBRARY
# ---------------------------------------------------------------------------

def test_family_calibration_and_prediction():
    assert len(F.class_subset(4)) == 4 and F.class_subset(4) == F.class_subset(4)
    a, b = F.fit_calibration([0.2, 0.4, 0.6], [0.4, 0.6, 0.8])
    assert a == approx(0.2) and b == approx(1.0)
    rib, acc = F.predict_rib(0.5, 10, 0.2, 1.0)
    assert acc == approx(0.7)
    assert rib == approx(C.task_info_fano(0.3, 10))            # the calibrated accuracy feeds the formula
    assert F.predict_rib(5.0, 10, 0.0, 1.0)[1] == 1.0                  # accuracy is clipped


def test_lib_verdict_blocks_over_exclusion():
    base = dict(c=10, acc_ridge=0.50, acc_randfeat=0.40, acc_t=0.80, rib=1.40, k_min=4,
                excl_measured=[2], excl_label_only=[2, 4], excl_pred=[2], rib_pred=1.35,
                acc_pred=0.79, k_min_pred=4)
    good = [dict(base), dict(base, c=6, acc_ridge=0.55, acc_t=0.83, rib=1.20, rib_pred=1.15),
            dict(base, c=3, acc_ridge=0.60, acc_t=0.86, rib=0.95, rib_pred=0.90)]
    assert F.lib_verdict(good)[0].startswith("SUPPORT")
    assert not F.lib_verdict([dict(base, excl_pred=[2, 4])])[0].startswith("SUPPORT")


# ---------------------------------------------------------------------------
# PRETRAINED WITNESS (the threshold table)
# ---------------------------------------------------------------------------

def test_witness_threshold_table_matches_the_measured_exclusion_set():
    th = {k: P.witness_threshold(k) for k in (2, 3, 4, 8)}
    assert th[2] == approx(0.577, abs=0.01)
    assert th[4] == approx(0.807, abs=0.01)
    assert th[4] > 0.803                                              # the CP-H teacher missed K=4 by a hair
    assert all(th[a] < th[b] for a, b in zip([2, 3, 4], [3, 4, 8]))
    assert P.witness_threshold(16) == float("inf")                     # K >= C is never excludable at C=10


def test_pw_verdict_requires_a_useful_pretrained_witness():
    rows = [dict(name="P0_raw_pixels_probe", acc=0.30, cost="", r_ib=1.0, k_min=3, excluded=[2],
                 coverage=0.0, false_excl=[], k_wall=128, looseness=4.0, fails=[2], succeeds=[128]),
            dict(name="P1_random_probe", acc=0.30, cost="", r_ib=1.0, k_min=3, excluded=[2],
                 coverage=0.0, false_excl=[], k_wall=128, looseness=4.0, fails=[2], succeeds=[128]),
            dict(name="P2_mnist_probe", acc=0.45, cost="", r_ib=1.0, k_min=3, excluded=[2],
                 coverage=0.0, false_excl=[], k_wall=128, looseness=4.0, fails=[2], succeeds=[128]),
            dict(name="P3_rot_probe", acc=0.72, cost="", r_ib=2.0, k_min=8, excluded=[2, 3],
                 coverage=0.5, false_excl=[], k_wall=128, looseness=2.5, fails=[2, 3, 4],
                 succeeds=[128])]
    assert P.pw_verdict(rows)[0].startswith("SUPPORT")
    wrong = [dict(r) for r in rows]
    wrong[3]["false_excl"] = [16]
    assert not P.pw_verdict(wrong)[0].startswith("SUPPORT")


# ---------------------------------------------------------------------------
# SCALE-C
# ---------------------------------------------------------------------------

def test_lattice_family_shapes_and_anchor():
    (xtr, ytr), (xte, yte) = S.make_lattice(4, n_per_class=3, seed=0)
    assert xtr.shape[1] == 3                                          # 3-dimensional lattice input
    assert len(torch.unique(torch.cat([ytr, yte]))) == 64             # the m=4 case is grid64
    assert len(ytr) == int(0.8 * (len(ytr) + len(yte)))               # the documented 80/20 split
    assert S.class_count(10) == 1000


def test_ols_helpers():
    a, b, r2 = S.ols1([1.0, 2.0, 3.0], [5.0, 7.0, 9.0])
    assert b == approx(2.0) and a == approx(3.0) and r2 == approx(1.0)
    assert S.ols1([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])[1] == 0.0          # a constant regressor is degenerate
    assert S.asymptote_note(2.0, 0.5) == approx(4.0)


def test_scale_verdict_scoring():
    base = dict(c=1000, seed=0, eps=0.20, rib=5.0, k_min=148, accs={}, rel={}, fails=[2],
                succeeds=[512], excluded=[2, 8, 32], k_wall=512, looseness=1.25, s1=True,
                false_excl=[], coverage=1.0)
    low = dict(base, c=10, eps=0.20, rib=1.5, k_min=5, excluded=[2], k_wall=8, looseness=2.6)
    mid = dict(base, c=100, eps=0.20, rib=3.0, k_min=20, excluded=[2, 8], k_wall=64, looseness=1.4)
    assert S.scale_verdict([low, mid, base])[0].startswith("SUPPORT")
    assert not S.scale_verdict([low, mid, dict(base, false_excl=[512])])[0].startswith("SUPPORT")
    assert not S.scale_verdict([low, mid, dict(base, excluded=[2])])[0].startswith("SUPPORT")
    no_wall = [dict(low, k_wall=None, looseness=None), dict(mid, k_wall=None, looseness=None),
               dict(base, k_wall=None, looseness=None)]
    assert not S.scale_verdict(no_wall)[0].startswith("SUPPORT")


def test_cached_target_kd_replica_matches_the_package_loop():
    """The third SCALE-C run's speed came from caching the teacher's logits; the claim that this is an
    exact replica of the package's KD objective is verified here numerically."""
    (x_tr, y_tr), _ = S.make_lattice(2, n_per_class=6, sigma=1.0, seed=0)
    ref = S.GridMLP(8, 2048)
    plain = DataLoader(TensorDataset(x_tr, y_tr), batch_size=16, shuffle=False)
    cached = DataLoader(TensorDataset(x_tr, y_tr, SC.teacher_logits(ref, plain, "cpu").float()),
                        batch_size=16, shuffle=False)
    torch.manual_seed(0)
    m1 = S.GridMLP(8, 8)
    l1 = C.train_one_epoch(m1, plain, torch.optim.Adam(m1.parameters(), lr=1e-2), device="cpu",
                           distill_from=ref, T_kd=3.0, beta=0.25)
    torch.manual_seed(0)
    m2 = S.GridMLP(8, 8)
    l2 = SC.train_one_epoch_cached(m2, cached, torch.optim.Adam(m2.parameters(), lr=1e-2), device="cpu")
    assert abs(l1 - l2) < 1e-4
    assert SC.K_TEACHER == 2048 and max(SC.K_GRID) == SC.K_TEACHER


# ---------------------------------------------------------------------------
# WALL-SATURATION (the Monte-Carlo ground truth and the mechanism criteria)
# ---------------------------------------------------------------------------

def test_lattice_ground_truth_is_monotone_in_noise():
    """The estimator that the whole mechanism argument rests on: less noise must mean less Bayes error and
    more information, and the scale-equivalence used by CROSS-TASK must show up in the geometry."""
    quiet = WS.lattice_posterior_stats(3, 0.35, n_mc=8000)
    noisy = WS.lattice_posterior_stats(3, 1.2, n_mc=8000)
    assert noisy["eps_bayes"] > quiet["eps_bayes"]
    assert noisy["mi"] < quiet["mi"]
    assert quiet["mi"] <= WS.math.log(27) + 1e-9                    # information cannot exceed ln C
    # sigma/spacing is the only combination that matters: doubling both leaves the task unchanged
    a = WS.lattice_posterior_stats(3, 0.5, spacing=3.0, n_mc=8000)
    b = WS.lattice_posterior_stats(3, 1.0, spacing=6.0, n_mc=8000)
    assert abs(a["mi"] - b["mi"]) < 0.05


def test_bayes_verdict_refutes_a_reference_far_from_the_ceiling():
    rows = [dict(step0=True, sigma=1.0, C=8, ref_eps=0.175, eps_bayes=0.187, k_min_bayes=4, k_wall=8),
            dict(step0=True, sigma=1.0, C=216, ref_eps=0.446, eps_bayes=0.299, k_min_bayes=24, k_wall=512)]
    v, _ = WS.bayes_verdict(rows, sigma=1.0)
    assert v.startswith("REFUTED")                                   # B1 fails: the witness is not at the ceiling
    good = [dict(r, ref_eps=r["eps_bayes"]) for r in rows]
    good[1]["k_wall"] = 8                                            # flat wall + flat floor = the M-BAYES shape
    good[0]["k_min_bayes"] = 8
    assert WS.bayes_verdict(good, sigma=1.0)[0].startswith("SUPPORT")


def test_sigma_verdict_requires_the_wall_to_respond():
    base = dict(sigma=1.0, eps=0.446, eps_bayes=0.299, k_wall=512, looseness_vs_bayes=1.97)
    low = dict(sigma=0.5, eps=0.045, eps_bayes=0.007, k_wall=None, looseness_vs_bayes=None)
    v, _ = WS.sigma_verdict([base, low])
    assert v.startswith("REFUTED")                                   # no located wall at the harder witness
    rises = [base, dict(low, k_wall=2048, looseness_vs_bayes=10.2)]
    assert WS.sigma_verdict(rises)[0].startswith("SUPPORT")
    assert WS.sigma_verdict([base, dict(low, k_wall=1024, looseness_vs_bayes=51.0)])[0].startswith("MIXED")


def test_lattice_centers_are_the_declared_grid():
    centers = WS.lattice_centers(3)
    assert centers.shape == (27, 3)
    assert float(centers.max()) == 6.0 and float(centers.min()) == 0.0


# ---------------------------------------------------------------------------
# CROSS-TASK (the P-A1-2 criteria) and SCALE-AXES
# ---------------------------------------------------------------------------

def test_cross_verdict_requires_invariance_and_ordering():
    b = dict(sigma=0.5, spacing=3.0, rib=4.95, mi_true=5.356, under_read_rib=0.40, k_min=142)
    c = dict(sigma=1.0, spacing=6.0, rib=4.78, mi_true=5.356, under_read_rib=0.58, k_min=119)
    a = dict(sigma=1.0, spacing=3.0, rib=2.29, mi_true=4.545, under_read_rib=0.87, k_min=10)
    v, _ = CT.cross_verdict([], [a, b, c])
    assert v.startswith("SUPPORT")
    # an inversion must fail X2 whenever the ground-truth difference is material, while leaving the
    # scale-equivalent pair untouched (otherwise the test would trip X1 instead and prove nothing)
    bad = [dict(a, rib=5.5), b, c]
    assert CT.cross_verdict([], bad)[0].startswith("MIXED")
    # breaking the invariance pair itself fails BOTH criteria (the low reading also inverts the order)
    assert CT.cross_verdict([], [a, b, dict(c, rib=1.0)])[0].startswith("REFUTED")


def test_scale_axes_criteria_are_direction_sensitive():
    assert SA.samples_verdict([dict(n_per_class=40, rib=2.24, under_read_rib=0.92, looseness_vs_bayes=None),
                               dict(n_per_class=300, rib=2.43, under_read_rib=0.73,
                                    looseness_vs_bayes=2.0)])[0].startswith("MIXED")
    # a clear width effect clears the robustness threshold; the marginal one (the measured data's shape) must
    # NOT be reported as support -- that is exactly what W2 was added for
    assert SA.width_verdict([dict(d=64, k_wall=2048, rel={512: 0.12}),
                             dict(d=256, k_wall=512, rel={512: 0.08})])[0].startswith("SUPPORT")
    assert SA.width_verdict([dict(d=64, k_wall=2048, rel={512: 0.103}),
                             dict(d=256, k_wall=512, rel={512: 0.081})])[0].startswith("MIXED/REFUTED")
    assert SA.resolution_verdict([dict(resolution=8, rib=4.159, k_wall=128),
                                  dict(resolution=16, rib=4.159, k_wall=128),
                                  dict(resolution=32, rib=4.159, k_wall=128)])[0].startswith("SUPPORT")


def test_resolution_resampling_preserves_the_class_structure():
    (x8, y8), _ = SA.make_grid64_at(8)
    (x32, y32), _ = SA.make_grid64_at(32)
    assert x8.shape[-1] == 8 and x32.shape[-1] == 32 and x8.shape[0] == x32.shape[0]
    assert len(torch.unique(y8)) == 64 and torch.equal(y8, y32)


# ---------------------------------------------------------------------------
# LOOSENESS-REAUDIT (the corrected D2' and its holdout)
# ---------------------------------------------------------------------------

def test_reaudit_ols_and_split():
    a, b, r2 = LR.ols([0.0, 1.0, 2.0], [1.0, 3.0, 5.0])
    assert abs(a - 1.0) < 1e-9 and abs(b - 2.0) < 1e-9 and abs(r2 - 1.0) < 1e-9
    rows = [dict(source="s", name=f"n{i}", eps=i / 10, looseness=1.0 + i, n_classes=10) for i in range(6)]
    fit, test = LR.mechanical_split(rows)
    assert len(fit) == 3 and len(test) == 3
    assert {r["name"] for r in fit} & {r["name"] for r in test} == set()      # a real partition


def test_reaudit_verdict_precedence():
    """A non-positive slope must refute regardless of how well a flat line 'predicts' -- that precedence is
    what the pre-registration says, and the first implementation got it wrong."""
    flat = [dict(source="s", name=f"n{i}", eps=i / 10, looseness=2.0, n_classes=10) for i in range(10)]
    v, _, nums = LR.reaudit(flat)
    assert v.startswith("REFUTED") and nums["b"] == 0.0
    exact = [dict(source="s", name=f"n{i}", eps=i / 10, looseness=0.5 + 14.0 * i / 10, n_classes=10)
             for i in range(10)]
    assert LR.reaudit(exact)[0].startswith("SUPPORT")


def test_reaudit_class_count_control_detects_a_spurious_regressor():
    """If the class count really drove looseness, C3 must fail; here it is built to."""
    rows = [dict(source="s", name=f"n{i:02d}", eps=0.1, looseness=float(i), n_classes=2 ** (i % 6))
            for i in range(12)]
    _, detail, nums = LR.reaudit(rows)
    assert nums["r2_class"] >= nums["r2"]
    assert any("C3" in line for line in detail)


# ---------------------------------------------------------------------------
# BUDGET-DOSE and FAMILY-B
# ---------------------------------------------------------------------------

def test_dose_power_law_fit_and_verdict():
    a, b, r2 = BD.fit_power([1, 2, 4, 8], [0.5 * x ** -0.5 for x in (1, 2, 4, 8)])
    assert abs(b + 0.5) < 1e-9 and abs(r2 - 1.0) < 1e-9 and abs(a - 0.5) < 1e-9
    # a clean shared exponent passes; a non-monotone instance must not
    good = ([dict(tag="I1", c=64, k=1024, epochs=e, rel=0.4 * e ** -0.5) for e in (60, 120, 240, 480)]
            + [dict(tag="I2", c=216, k=4096, epochs=e, rel=0.9 * e ** -0.5) for e in (60, 180, 360)])
    assert BD.dose_verdict(good)[0].startswith("SUPPORT")
    nonmono = ([dict(tag="I1", c=64, k=1024, epochs=e, rel=0.4 * e ** -0.5) for e in (60, 120, 240, 480)]
               + [dict(tag="I2", c=216, k=4096, epochs=e, rel=r) for e, r in ((60, 0.12), (180, 0.068), (360, 0.109))])
    v, detail, _ = BD.dose_verdict(nonmono)
    assert not v.startswith("SUPPORT")
    assert any("I2" in line and "FAIL" in line for line in detail)


def test_family_b_ground_truth_and_criteria():
    centers = FB.random_centers(8, 3.0)
    assert torch.equal(centers, FB.random_centers(8, 3.0))                  # reproducible draw
    a = FB.mixture_stats(centers, 0.5, n_mc=8000)
    b = FB.mixture_stats(centers, 2.0, n_mc=8000)
    assert b["eps_bayes"] > a["eps_bayes"] and b["mi"] < a["mi"]
    # exact scale equivalence: centres and noise scaled together leave every posterior unchanged
    scaled = FB.mixture_stats(centers * 2.0, 1.0, n_mc=8000)
    assert abs(a["mi"] - scaled["mi"]) < 0.05
    y1, y2, _ = FB.family_verdict([dict(tag="B1", rib=2.67, mi_true=4.60, rib_bayes=3.21),
                                   dict(tag="B2", rib=0.98, mi_true=3.27, rib_bayes=1.26),
                                   dict(tag="B3", rib=2.61, mi_true=4.60, rib_bayes=3.21)])
    assert y1 and y2
    assert not FB.family_verdict([dict(tag="B1", rib=2.67, mi_true=4.60, rib_bayes=3.21),
                                  dict(tag="B2", rib=0.98, mi_true=3.27, rib_bayes=1.26),
                                  dict(tag="B3", rib=0.10, mi_true=4.60, rib_bayes=3.21)])[0]


def test_family_b_geometry_note_reports_the_irregularity():
    centers = FB.random_centers(216, 3.0)
    g = FB.geometry_note(centers)
    assert g["nn_min"] < g["nn_median"] < g["nn_max"]                # a real spread, not a lattice
    assert g["nn_min"] > 0


# ---------------------------------------------------------------------------
# BUDGET-MECHANISM (the recipe attribution)
# ---------------------------------------------------------------------------

def test_mechanism_monotonicity_and_attribution():
    assert not BM.is_monotone_decreasing([dict(epochs=e, rel=r) for e, r in ((60, 0.12), (180, 0.068), (360, 0.109))])
    assert BM.is_monotone_decreasing([dict(epochs=e, rel=r) for e, r in ((60, 0.12), (180, 0.09), (360, 0.07))])
    # a factor that restores monotonicity is credited
    rows = ([dict(recipe=BM.recipe_name("fixed", True), schedule="fixed", restarts=True, epochs=e, rel=r)
             for e, r in ((60, 0.1207), (180, 0.0685), (360, 0.1087))]
            + [dict(recipe=BM.recipe_name("cosine", True), schedule="cosine", restarts=True, epochs=e, rel=r)
               for e, r in ((60, -0.0076), (180, -0.0158), (360, -0.0189))])
    v, detail = BM.mechanism_verdict(rows)
    assert v.startswith("SUPPORT") and "fixed learning rate" in v
    assert any("R1" in line and "PASS" in line for line in detail)
    # nothing restored -> MIXED, not SUPPORT
    flat = [dict(r, rel=0.1) for r in rows]
    assert BM.mechanism_verdict(flat)[0].startswith("REFUTED")      # no non-monotonicity to explain


def test_mechanism_recipe_names_and_knob():
    assert BM.recipe_name("fixed", True) == "fixed_restartson"
    assert BM.recipe_name("cosine", False) == "cosine_restartsoff"
    # the restart knob must be wired into the loop that is verified against the replica
    (x_tr, y_tr), _ = S.make_lattice(2, n_per_class=4, sigma=1.0, seed=0)
    ref = S.GridMLP(8, 2048)
    plain = DataLoader(TensorDataset(x_tr, y_tr), batch_size=8, shuffle=False)
    tgt = SC.teacher_logits(ref, plain, "cpu").float()
    loader = DataLoader(TensorDataset(x_tr, y_tr, tgt), batch_size=8, shuffle=False)
    torch.manual_seed(0)
    m1 = S.GridMLP(8, 8)
    l_on = BM.train_epoch_recipe(m1, loader, torch.optim.Adam(m1.parameters(), lr=1e-2), "fixed", True, 0, 1)
    torch.manual_seed(0)
    m2 = S.GridMLP(8, 8)
    l_off = BM.train_epoch_recipe(m2, loader, torch.optim.Adam(m2.parameters(), lr=1e-2), "fixed", False, 0, 1)
    assert abs(l_on - l_off) > 1e-9                                 # the knob changes something
    torch.manual_seed(0)
    m3 = S.GridMLP(8, 8)
    l_ref = SC.train_one_epoch_cached(m3, loader, torch.optim.Adam(m3.parameters(), lr=1e-2))
    assert abs(l_on - l_ref) < 1e-6                                 # default cell == verified replica


# ---------------------------------------------------------------------------
# WALL-REMEASURE (shape survives, the drop expectation does not)
# ---------------------------------------------------------------------------

def test_remeasure_verdict_separates_shape_from_expectation():
    rows = [dict(arm="scale_c_cosine", c=8, k_wall=32, old_k_wall=8, looseness=2.7, false_excl=[]),
            dict(arm="scale_c_cosine", c=27, k_wall=128, old_k_wall=128, looseness=2.8, false_excl=[]),
            dict(arm="scale_c_cosine", c=64, k_wall=512, old_k_wall=512, looseness=3.0, false_excl=[]),
            dict(arm="scale_c_cosine", c=216, k_wall=512, old_k_wall=512, looseness=2.7, false_excl=[])]
    v, detail, shape = WR.remeasure_verdict(rows)
    # the measured outcome: the drop failed, the shape held, soundness held
    assert "REFUTED for the expectation" in v and "CONFIRMED for the shape" in v
    assert "saturating at the top: True" in shape and "non-monotone: True" in shape
    assert any("S3" in line and "PASS" in line for line in detail)
    # a broken soundness criterion must dominate, whatever the shape says
    bad = [dict(r, false_excl=[64]) if r["c"] == 64 else r for r in rows]
    assert WR.remeasure_verdict(bad)[0].startswith("REFUTED: the re-measurement breaks the soundness")
    # and a genuine drop with a surviving shape is SUPPORT
    good = [dict(r, k_wall=max(2, r["old_k_wall"] // 2) if r["c"] == 8 else r["k_wall"]) for r in rows]
    assert WR.remeasure_verdict(good)[0].startswith("SUPPORT")


def test_remeasure_schedule_is_wired_and_replica_faithful():
    """The no-scheduler path must equal the verified replica; the cosine path must actually differ."""
    (x_tr, y_tr), _ = S.make_lattice(2, n_per_class=5, sigma=1.0, seed=0)
    ref = S.GridMLP(8, 2048)
    plain = DataLoader(TensorDataset(x_tr, y_tr), batch_size=8, shuffle=False)
    tgt = SC.teacher_logits(ref, plain, "cpu").float()
    loader = DataLoader(TensorDataset(x_tr, y_tr, tgt), batch_size=8, shuffle=False)
    torch.manual_seed(0)
    a = S.GridMLP(8, 8)
    WR.train_epoch_cosine(a, loader, torch.optim.Adam(a.parameters(), lr=1e-2), None)
    torch.manual_seed(0)
    b = S.GridMLP(8, 8)
    SC.train_one_epoch_cached(b, loader, torch.optim.Adam(b.parameters(), lr=1e-2))
    assert torch.allclose(a.fc2.weight, b.fc2.weight, atol=1e-7)
    torch.manual_seed(0)
    c = S.GridMLP(8, 8)
    opt = torch.optim.Adam(c.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4)
    for _ in range(2):
        WR.train_epoch_cosine(c, loader, opt, sched)
    assert opt.param_groups[0]["lr"] < 1e-2                        # the schedule really moved the LR
    assert not torch.allclose(a.fc2.weight, c.fc2.weight, atol=1e-9)


def test_cph_probe_verdict_flags_a_material_recipe_effect():
    """The real-data probe's criterion: a paired difference >= 0.03 counts as 'the recipe matters'."""
    def rows(fixed, cosine, recorded=None):
        out = []
        for k, f, c in zip((64, 128, 192), fixed, cosine):
            out.append(dict(arm="cph_probe", k=k, recipe="fixed", rel=f, acc_t=0.799,
                            rel_recorded_120ep=recorded))
            out.append(dict(arm="cph_probe", k=k, recipe="cosine", rel=c, acc_t=0.799,
                            rel_recorded_120ep=recorded))
        return out
    v, detail = WR.cph_verdict(rows((0.324, 0.275, 0.171), (0.207, 0.189, 0.140), 0.090))
    assert v.startswith("MIXED") and any("MATTERS" in line for line in detail)
    v2, _ = WR.cph_verdict(rows((0.20, 0.19, 0.14), (0.205, 0.192, 0.138)))
    assert v2.startswith("SUPPORT")                                 # < 0.03 everywhere -> recipe-robust
    assert WR.cph_verdict([])[0].startswith("UNDECIDABLE")


# ---------------------------------------------------------------------------
# FAMILY-C (16-D) and the non-degenerate resolution axis
# ---------------------------------------------------------------------------

def test_family_c_ground_truth_and_criteria():
    centers = FC.random_centers(n_classes=8, spread=1.5)
    assert centers.shape == (8, 16)                                  # the family's defining property
    a = FC.mixture_stats(centers, 1.0, n_mc=8000)
    b = FC.mixture_stats(centers, 4.0, n_mc=8000)
    assert b["eps_bayes"] > a["eps_bayes"] and b["mi"] < a["mi"]
    scaled = FC.mixture_stats(centers * 2.0, 2.0, n_mc=8000)         # the exact twin, in 16 dimensions
    assert abs(a["mi"] - scaled["mi"]) < 0.05
    # the measured outcome: the twin pair disagreed by 0.429 nats -> the invariance criterion fails
    measured = [dict(tag="C1", rib=3.963, mi_true=5.243, rib_bayes=4.966),
                dict(tag="C2", rib=3.534, mi_true=5.243, rib_bayes=4.966),
                dict(tag="C3", rib=1.919, mi_true=4.884, rib_bayes=1.706)]
    v, detail = FC.family_verdict(measured, others=[dict(family="A (3-D lattice)", mi=5.161, under=0.802,
                                                         tag="lattice sigma=0.70")])
    assert v.startswith("REFUTED")
    assert any("Z1" in line and "FAIL" in line for line in detail)
    assert any("Z2" in line and "PASS" in line for line in detail)
    # and a genuinely invariant pair is still credited
    ok_rows = [dict(r, rib=3.96 if r["tag"] == "C2" else r["rib"]) for r in measured]
    assert not FC.family_verdict(ok_rows)[0].startswith("REFUTED")


def test_resolution_hard_criteria_require_a_non_degenerate_reading():
    """R3' is the criterion the first A3 run would have failed: the reading must be off the ceiling."""
    rows = [dict(resolution=8, rib=1.246, k_wall=128), dict(resolution=16, rib=1.532, k_wall=128),
            dict(resolution=32, rib=1.444, k_wall=128)]
    v, detail = SA.resolution_hard_verdict(rows)
    assert v.startswith("SUPPORT") and any("R3'" in line and "PASS" in line for line in detail)
    saturated = [dict(r, rib=4.159) for r in rows]                   # the saturated point the first run hit
    v2, detail2 = SA.resolution_hard_verdict(saturated)
    assert not v2.startswith("SUPPORT")
    assert any("R3'" in line and "FAIL" in line for line in detail2)
    assert SA.resolution_hard_verdict([])[0].startswith("UNDECIDABLE")


def test_saturation_probe_verdict_judges_growth_not_saturation():
    """The measured outcome: four class counts on fine ladders give walls 128/256/512/768 -> C^0.60."""
    rows = [dict(arm="saturation_probe", c=27, k_wall=128,
                 rel={16: 0.395, 96: 0.111, 128: 0.089, 192: 0.060}),
            dict(arm="saturation_probe", c=64, k_wall=256,
                 rel={128: 0.285, 192: 0.179, 256: 0.089, 512: 0.052}),
            dict(arm="saturation_probe", c=216, k_wall=512,
                 rel={256: 0.190, 384: 0.150, 512: 0.015, 1024: -0.053}),
            dict(arm="saturation_probe", c=512, k_wall=768,
                 rel={512: 0.123, 768: 0.042, 1024: 0.003, 2048: -0.113})]
    v, detail = WR.saturation_verdict(rows, fit_rows=rows)
    assert v.startswith("REFUTED for the saturation claim")
    assert "C^0.60" in v or "C^0.6" in v                                          # the fitted exponent
    assert any("the wall grows at the same rate as the requirement" in line for line in detail)
    assert any("SOFT boundary" in line for line in detail)                        # C=27's drop is 0.021
    # a genuinely shared wall (no growth) is still reported as such
    shared = [dict(arm="saturation_probe", c=64, k_wall=512, rel={128: 0.28, 256: 0.15, 512: 0.09}),
              dict(arm="saturation_probe", c=216, k_wall=512, rel={256: 0.30, 512: 0.09, 1024: 0.05})]
    assert WR.saturation_verdict(shared, fit_rows=shared)[0].startswith("SUPPORT (a real shared boundary)")
