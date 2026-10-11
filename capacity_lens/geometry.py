"""Separability measures: two families of pure functions (predictive indicators of cliff localizability / decision margins in function space).

Background (CP1 experiment; see falsified.md):
- Function B v3 (MLP grid64, d=3 discrete-cluster representation) distillation cliff is **localizable** (K*=96/96/96);
- C3 v3 (CNN grid64, d=64 continuous representation) cliff is **not localizable** (K* 0/3, the drop decays
  smoothly with log K).

This module quantifies the "representation cluster structure" into a metric comparable across arms, used to predict,
**before** the distillation experiment, whether the cliff is localizable -- upgrading a negative result from "recording a phenomenon" to "a predictable criterion".

Second family (decision margins in function space, CP-D experiment; see falsified.md):
CP1 and CP3 supplementary test ① confirmed that **static representation readings** (silhouette / codebook margin / I(Z;Y) endpoint /
rate-distortion curve) all fail to distinguish true differences after acc saturates (two CNN grids had I(Z;Y)=4.033+-0.000
exactly equal while rho differed by 50%). So we switch to the language of **detection theory**: treat the "true class" as the signal and the "strongest alternative class"
as the competing hypothesis, and measure the ratio of the scale of the decision margin to the noise floor -- this is a quantity that still has resolving power after acc reaches
the ceiling.

**Final verdict for this family (CP-D, 2026-09-26): judged negative, not part of the public API.** The decision margin d' is indeed
correlated with training difficulty (3/3 seeds Spearman <= -0.68 on shape64), but it is **no better than the cheapest
acc@10 (0.77)**, and does not replicate at all on grid64 (|rho| <= 0.29) -- by the pre-registered J3/J4/J5
it is judged an honest negative result. Hence the functions in this family are kept only for **research use** (to reproduce the negative result and to help later work avoid the pitfall),
and `capacity_lens/__init__.py` **does not export** them; externally they are positioned as "the 6th falsified discriminability candidate
reading". See the CP-D record in docs/falsified.md and Section 4.3 of docs/framework.md.
"""

import math

import torch
import torch.nn.functional as F


def _pairwise_dist(feats, metric):
    """Pairwise distance matrix [N, N] (internal utility, not an exported interface)."""
    if metric == "euclid":
        return torch.cdist(feats, feats)
    if metric == "cosine":
        f = F.normalize(feats, dim=-1)
        return (1.0 - f @ f.t()).clamp_(0.0, 2.0)
    raise ValueError(f"unknown metric: {metric!r} (supported: 'euclid' / 'cosine')")


def _check_inputs(feats, labels):
    """Input validation (internal utility): returns (feats[N,d] float, labels[N] long)."""
    feats = torch.as_tensor(feats).float()
    labels = torch.as_tensor(labels).long().reshape(-1)
    if feats.dim() != 2:
        raise ValueError(f"feats must be [N, d], got shape {tuple(feats.shape)}")
    if feats.shape[0] != labels.shape[0]:
        raise ValueError(f"feats and labels sample counts differ: {feats.shape[0]} vs {labels.shape[0]}")
    if feats.shape[0] < 2:
        raise ValueError("N < 2, metric undefined")
    if feats.shape[0] > 20000:
        raise ValueError("N > 20000 (pairwise distance matrix memory guard; this tool targets test sets with N~2e3)")
    if torch.unique(labels).numel() < 2:
        raise ValueError("fewer than 2 effective classes, metric undefined")
    return feats, labels


def silhouette_by_label(feats, labels, metric="euclid"):
    """Average silhouette coefficient by class label (pure function, independently unit-testable).

    s(i) = (b(i) - a(i)) / max(a(i), b(i)), where a(i) = the average distance from i to
    other points of the same class, and b(i) = the average distance from i to the **nearest other class**. Range [-1, 1]:
    close to 1 = class clusters clearly separated; close to 0 = class overlap (continuous representation); negative = classes interleaved.

    Why choose it as the main cliff-prediction indicator: the VQ bottleneck assigns samples to the same codeword by nearest neighbor,
    and the silhouette's "same-class distance < other-class distance" is exactly the structural condition under which the quantization partition preserves class information;
    it is also bounded in [-1,1], so the euclid/cosine arm values are comparable across arms (quantities such as the separation ratio in raw units
    drift with dimension and metric).

    Boundary conventions:
    - single-sample class: a(i) is undefined, and following the sklearn convention this point has s(i) = 0;
    - max(a, b) == 0 (coincident with all points): s(i) = 0;
    - N < 2 / number of classes < 2 / N > 20000 / unknown metric -> ValueError.
    """
    feats, labels = _check_inputs(feats, labels)
    n = feats.shape[0]
    d = _pairwise_dist(feats, metric)                       # [N, N]

    n_classes = int(labels.max().item()) + 1
    onehot = F.one_hot(labels, num_classes=n_classes).float()   # [N, C]
    counts = onehot.sum(dim=0)                              # [C]
    class_sums = d @ onehot                                 # [N, C] row i col k = sum of distances from i to all points of class k

    # a(i): within-class mean distance (single-sample class -> 0, then set s=0 overall)
    cnt_own = counts[labels]                                # [N]
    own_sum = class_sums.gather(1, labels[:, None]).squeeze(1)
    a = torch.where(cnt_own > 1,
                    own_sum / (cnt_own - 1).clamp(min=1),
                    torch.zeros(n))

    # b(i): mean distance to the nearest other class (empty classes and own class excluded)
    mean_dist = class_sums / counts.clamp(min=1).unsqueeze(0)
    mean_dist[:, counts == 0] = float("inf")
    mean_dist[onehot.bool()] = float("inf")
    b = mean_dist.min(dim=1).values                         # [N]

    denom = torch.maximum(a, b)
    s = torch.where(denom > 0, (b - a) / denom, torch.zeros(n))
    s = torch.where(cnt_own == 1, torch.zeros(n), s)        # single-sample class: sklearn convention
    return s.mean().item()


def separation_ratio(feats, labels, metric="euclid"):
    """Minimum inter-class-center distance / average within-class spread (pure function, independently unit-testable; only for mechanism explanation, not part of the verdict).

    Physical meaning: "how many within-class scales apart are the two nearest class centers". Roughly >= 3 corresponds to a clear cluster
    "separated by several sigma", and ~1 corresponds to a continuous overlapping manifold. Division of labor with silhouette: this indicator
    has an intuitive physical picture but is not comparable across dimension/metric, so it is only printed, not used for the verdict.
    """
    feats, labels = _check_inputs(feats, labels)
    classes = torch.unique(labels)
    centroids = torch.stack([feats[labels == k].mean(dim=0) for k in classes])

    if metric == "euclid":
        between = torch.cdist(centroids, centroids)
        within_d = (feats - centroids[torch.searchsorted(classes, labels)]).norm(dim=1)
        within = within_d.mean().item()
    elif metric == "cosine":
        cn = F.normalize(centroids, dim=-1)
        between = 1.0 - cn @ cn.t()
        fn = F.normalize(feats, dim=-1)
        own = cn[torch.searchsorted(classes, labels)]
        within = (1.0 - (fn * own).sum(dim=1)).clamp(0.0, 2.0).mean().item()
    else:
        raise ValueError(f"unknown metric: {metric!r}")

    eye = torch.eye(between.shape[0], dtype=torch.bool)
    min_between = between[~eye].min().item()
    if within <= 0:
        return float("inf")
    return min_between / within


def vq_quantization_error(feats, codebook, metric="euclid"):
    """Relative quantization error under the teacher's own codebook (pure function, independently unit-testable; only for mechanism explanation, not part of the verdict).

    A direct measurement of the "near-lossless region": if the quantization error under the K=256 codebook is still of the same order as the within-class scale,
    it means the representation has no near-lossless region -- the mechanistic attribution of the C3 negative result.

    - euclid: mean(||t - nearest code||) / mean(||t||) (relative reconstruction error);
    - cosine: mean(1 - cos(t, nearest code)) (the nearest code is selected by cosine, and the error is also reported by cosine).
    """
    feats = torch.as_tensor(feats).float()
    codebook = torch.as_tensor(codebook).float()
    if feats.dim() != 2 or codebook.dim() != 2:
        raise ValueError("feats and codebook must both be 2-D [N, d] / [K, d]")
    if feats.shape[0] == 0 or codebook.shape[0] == 0:
        raise ValueError("empty input")

    if metric == "euclid":
        idx = torch.cdist(feats, codebook).argmin(dim=1)
        err = (feats - codebook[idx]).norm(dim=1).mean().item()
        scale = feats.norm(dim=1).mean().item()
        if scale <= 0:
            return float("inf")
        return err / scale
    if metric == "cosine":
        fn = F.normalize(feats, dim=-1)
        cn = F.normalize(codebook, dim=-1)
        best = (fn @ cn.t()).max(dim=1).values
        return (1.0 - best).clamp_(0.0, 2.0).mean().item()
    raise ValueError(f"unknown metric: {metric!r} (supported: 'euclid' / 'cosine')")


# ---------------------------------------------------------------------------
# Second family: decision margins in function space (detection-theory formulation; for the CP-D experiment; pure functions, independently unit-testable)
# ---------------------------------------------------------------------------

def decision_margins(scores, labels, mode="logit"):
    """Per-sample decision margin m(x) = s(y_true) - max_{j != y_true} s(j) (pure function).

    Detection-theory formulation: the true class is the "signal", the **strongest alternative class** is the "competing hypothesis", and m is the likelihood
    advantage of this sample over the hardest competing hypothesis. m <= 0 is strictly equivalent to misclassification (argmax is not the true class).

    mode="logit" (**default, primary formulation**): take the difference directly on the logits (log-odds margin).
        Analogous to the log-likelihood ratio; unbounded but **non-saturating** -- it still has resolving power after acc reaches the ceiling.
    mode="prob" (**control, not part of the verdict**): apply softmax first, then take the probability difference, range [-1, 1].
        It looks "bounded and therefore safer", but in fact at acc~0.97 p_true has already saturated to ~1.0000,
        and the median absolute deviation MAD degenerates to 0 -> ratio-type indicators (d') become undefined or blow up.
        Confirmed by measurement in stage 0 on 2026-09-26 (14 of 54 units with noise=0), so it is kept as a control.

    Scale issue: the logit scale drifts with the configuration (width/lr/codebook), so absolute values are not comparable across configurations;
    but the d' of detection_index is normalized by the **configuration's own MAD**, so d' is a scale-invariant quantity,
    which exactly cancels the logit drift.
    """
    s = torch.as_tensor(scores).float()
    y = torch.as_tensor(labels).long().reshape(-1)
    if s.dim() != 2:
        raise ValueError(f"scores must be [N, C], got shape {tuple(s.shape)}")
    if s.shape[1] < 2:
        raise ValueError("C < 2 classes, decision margin undefined")
    if s.shape[0] != y.shape[0]:
        raise ValueError(f"scores and labels sample counts differ: {s.shape[0]} vs {y.shape[0]}")
    if int(y.min()) < 0 or int(y.max()) >= s.shape[1]:
        raise ValueError("labels out of range (must be within [0, C))")
    if mode == "prob":
        s = torch.softmax(s, dim=1)
    elif mode != "logit":
        raise ValueError(f"unknown mode: {mode!r} (supported: 'logit' / 'prob')")
    s_true = s.gather(1, y[:, None]).squeeze(1)
    s_other = s.clone()
    s_other.scatter_(1, y[:, None], float("-inf"))
    return s_true - s_other.max(dim=1).values


def _quantile_sorted(sorted_v, p):
    """The p-quantile of a sorted tensor (order-statistic definition; does not rely on the version-dependent behavior of torch.quantile)."""
    n = sorted_v.numel()
    idx = min(n - 1, max(0, int(math.floor(p * (n - 1)))))
    return sorted_v[idx].item()


def detection_index(margins, labels, low_q=0.05):
    """Detection-theory discriminability metric group (pure function; primary metric = d = s_low / noise).

    Returns dict:
      s_low    = Q_low(m)                             -- low-tail margin (hard-sample end)
      s_med    = median_x m(x)                        -- median margin over all samples (auxiliary)
      s_worst  = min_c median_{x in class c} m(x)     -- median of the weakest class (**recorded only**)
      noise    = MAD_x m(x)                           -- robust noise floor (median absolute deviation)
      d        = s_low / noise                        -- **primary metric** (scale-invariant, of the form d')
      d_med    = s_med / noise                        -- auxiliary
      d_worst  = s_worst / noise                      -- auxiliary
      err      = frac(m <= 0)                         -- for reconciliation with 1 - acc
      noise_zero = bool                               -- degeneracy flag for noise == 0

    Why s_low uses a **quantile** rather than min_c(class median): the latter is the extremum of 64 class medians,
    and in an era where each class has only ~19-42 val samples it is an extreme order statistic that jumps wildly across seeds in measurement
    (same configuration s0/s1/s2 = -1.000 / -1.000 / +0.985, see the stage-0 log of 2026-09-26),
    carrying "which class collapsed this time" rather than a stable property of the configuration. The quantile is the robust version of the same
    intent (looking at the hard-sample end). The original min_c version is kept as s_worst for record only.

    Why s_med is not the primary metric: the multi-class error rate is determined by the hardest samples, and the global median is inflated by easy samples,
    losing resolving power after acc saturates -- the same pathology as the I(Z;Y) endpoint.
    """
    m = torch.as_tensor(margins).float().reshape(-1)
    y = torch.as_tensor(labels).long().reshape(-1)
    if m.shape[0] != y.shape[0]:
        raise ValueError(f"margins and labels sample counts differ: {m.shape[0]} vs {y.shape[0]}")
    if m.shape[0] < 2:
        raise ValueError("N < 2, metric undefined")
    classes = torch.unique(y)
    if classes.numel() < 2:
        raise ValueError("fewer than 2 effective classes, metric undefined")
    sorted_m, _ = torch.sort(m)
    med = _quantile_sorted(sorted_m, 0.5)
    s_low = _quantile_sorted(sorted_m, low_q)
    s_worst = min(m[y == int(c)].median().item() for c in classes)
    noise = (m - med).abs().median().item()
    nz = (noise <= 0.0)

    def _r(num):
        return None if nz else num / noise

    return dict(s_low=s_low, s_med=med, s_worst=s_worst, noise=noise,
                d=_r(s_low), d_med=_r(med), d_worst=_r(s_worst),
                err=float((m <= 0).float().mean().item()),
                noise_zero=bool(nz), n=int(m.shape[0]),
                n_classes=int(classes.numel()))


# ---------------------------------------------------------------------------
# Wall-geometry probe (task applicability pre-check; the P-A1-4 family)
# ---------------------------------------------------------------------------

DISCRETE_THRESHOLD = 0.02   # calibrated by the P-A1-4 dose response (pa4_demo, 2026-09-30):
# err_1nn was 0.000 at delta=0 (discrete) vs >=0.114 at delta=0.15+ (diffuse), cleanly straddling 0.02


def err_1nn(feats, labels, n_sub=3000, device="cpu"):
    """1-NN error on the first n_sub samples (subsampled pairwise, memory-guarded; pure function).

    The saturation gauge of the representation: 0 means trivially separable (the synthetic hard-wall
    regime; CP-E measured 0 on all 20 units), >>0 means a diffuse real-task representation.
    """
    f = feats[:n_sub].to(device)
    lab = labels[:n_sub].to(device)
    dist = torch.cdist(f, f)
    dist.fill_diagonal_(float("inf"))
    pred = lab[dist.argmin(dim=1)]
    return float((pred != lab).float().mean().item())


def wall_geometry_probe(feats, labels, metric="euclid", n_sub=3000, device="cpu"):
    """Task wall-geometry classification (pure function; the pre-flight "which half applies" check).

    Grounding (CP-H / P-A1-4, 2026-09-30): the Fano exclusion transfers to real tasks, but cliff (K*)
    location does not -- cliff SHARPNESS is a function of the discreteness of the output mapping, not of
    capacity. This probe measures that discreteness on a reference model's representation and classifies
    the task:

      "discrete" : err_1nn <= DISCRETE_THRESHOLD -> the output mapping is near-discrete; the K* locator
                   is applicable (the synthetic hard-wall regime: grid64 had err_1nn = 0);
      "diffuse"  : otherwise -> the mapping is diffuse; only the Fano exclusion half is promised,
                   K* location is NOT (CIFAR-10 measured err_1nn = 0.23).

    DISCRETE_THRESHOLD = 0.02 is CALIBRATED by the P-A1-4 dose-response experiment (pa4_demo,
    2026-09-30): the delta ladder measured err_1nn = 0.000 at delta=0 vs >= 0.114 at delta>=0.15,
    and locatability 0.67 -> 0.33 -> 0.33 -> 0.00 with sharpness 0.62 -> 0.60 -> 0.53 -> 0.40 --
    cliff sharpness is a function of output-mapping discreteness, the Fano exclusion is not.

    Returns a dict with the readings and the classification; never a verdict on quality -- only which
    half of the tool applies to this task.
    """
    feats, labels = _check_inputs(feats, labels)
    e1 = err_1nn(feats, labels, n_sub=n_sub, device=device)
    sil = silhouette_by_label(feats[:n_sub], labels[:n_sub], metric=metric)
    sep = separation_ratio(feats[:n_sub], labels[:n_sub], metric=metric)
    kind = "discrete" if e1 <= DISCRETE_THRESHOLD else "diffuse"
    return dict(err_1nn=e1, silhouette=sil, separation=sep, wall=kind,
                k_star_locatable=(kind == "discrete"),
                exclusion_applicable=True,
                note=("discrete wall: K* location applies (threshold calibrated by P-A1-4)"
                      if kind == "discrete" else
                      "diffuse wall: only exclusion is promised (K_min + certified screening), "
                      "NOT K* location"))
