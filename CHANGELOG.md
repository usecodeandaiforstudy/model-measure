# Changelog

All notable changes to the `capacity_lens` package. Versions are bumped in-tree; **nothing has been
published to PyPI**, so treats version numbers as development milestones rather than releases.

## 0.7.0 — 2026-10-15

### Added
- `capacity_lens/compress.py`: the compression budget advisor. Certified side: `task_info_floor`
  (Fano, with the empirical `h_y` and a `balance_warning` when the balanced-class default is invalid),
  `capacity_floors` (codebook / capacity / per-dimension bit floors), `veto` (the necessary-only
  `capacity < R_IB` gate) and `tightness_interval`, which brackets the true task-relevant looseness
  between `I_lb / R_IB` and `H(y) / R_IB` — registered at **≤1.253×** (MNIST) and **≤1.917×**
  (CIFAR-10), i.e. the certificate is provably tight in matched units. Measured side, zero training:
  `partition_retention` (hard quantisation keeps **0.15–0.75** of `I(z;y)` at K=64, always below 1) and
  `partition_ceiling` (**2–3× loose**, sound in direction only).
- `python -m capacity_lens.compress`: the certified report as a command line (`--selftest` included);
  the measured sections need a trained model and stay in the campaign driver.
- `docs/compression-advisor.md` (+ the `_cn` twin): the module's evidence grades — **[CERT] / [MEAS] /
  [ARCH] / [LOOSE]** — and the boundaries it refuses to hide (the floor sits 10–30× below the practical
  knee; rank starvation is irreversible, bit starvation is QAT-recoverable; allocate only by
  whole-network evaluation on a validation split; every wall is protocol-relative).
- `tests/test_compress_advisor.py`: pins the registered numbers (`R_IB` 1.8367 / 1.2009, the two
  brackets, the veto boundary, the degenerate bit floor on MNIST) so a change in the arithmetic shows
  up in CI rather than in a report.

## 0.6.1 — 2026-10-05

### Added
- `k_wall_band(k_list, accs, acc_ref, rel_tols=(0.05, 0.10, 0.15))`: the practical wall at several retention
  tolerances. The wall is a threshold crossing, so on a flat gap curve its value is partly the criterion rather
  than the capacity — measured: a 2-point retention difference moved a located wall from 2048 to 512
  (`docs/scale-axes.md`, A2). `preflight_report` now returns the band (`wall_band`), exposes it in
  `items` (`wall_band`, `wall_band_wide`) and states it in the verdict when the band swings by a factor of 4 or
  more, because "quote it as an interval, not a number" is the honest form of a sensitivity-sensitive answer.

## 0.6.0 — 2026-09-30

### Added
- `excluded_set`, `coverage`, `false_exclusions` are now part of the package. They are the product's
  headline rule (`log K < R_IB` ⇒ the bottleneck provably cannot carry the task) and were previously
  reachable only from the experiment scripts; a rule that ships as a *product* belongs in the library.
- `pyproject.toml`: the library is pip-installable (`pip install -e .`), the version is read dynamically
  from `capacity_lens.__version__`, and the built wheel contains **only** `capacity_lens/` plus the
  license — `experiments/` (the evidence) and `reference/` (the documentation) stay in the repository.
- `tests/`: 71 cases covering the library's implementation invariants and published rules, plus the
  pre-registered criterion functions of the experiment scripts. Runs with pytest **or**
  `python tests/run_all.py` (the verified GPU environment has torch but no pytest).

### Removed
- `TrainMonitor`'s `overfit` flag (`H(Z) > R_IB`) and the `task_mi_estimate` argument of `step()`. The
  direction is unsound, not merely unverified: `H(Z)` is a representation-entropy **upper** bound on
  `I(X;T)` and systematically overestimates the task information (measured 5.36 nats vs the task's true
  4.16), so the comparison fires on the estimator's bias. Capacity-vs-task questions are decided by the
  Fano exclusion direction instead. An unverified verdict-shaped API is worse than no API.

### Fixed
- `GaussBottleneck` is now an actual channel: the representation is normalized to unit RMS (so the SNR,
  hence the nominal capacity, is defined), the noise is active at **train and eval** time (the earlier
  version applied it only while training, i.e. at evaluation it was not a channel at all), `capacity()`
  reports the Shannon bound, and `forward` follows the package's `(Z, idx)` convention with `idx = None`.
  A `normalize="per_dim"` mode is available when a per-dimension variance bound is required.
- `evaluate` tolerates a bottleneck that reports no discrete codes (`info = [None]`): a model written as
  `return self.fc2(z), [idx]` now works with either a `VQBottleneck` or a `GaussBottleneck`.
- `TrainMonitor.summary()` labels its quantities as **nats** (with the bit equivalent in parentheses); the
  earlier version printed "bits" over natural-log values.

## 0.5.0 — 2026-09-30

### Added
- `measured_wall(k_list, accs, acc_ref, rel_tol)`: the practical wall, i.e. the smallest `K` whose student
  retains (1 − rel_tol) of the reference accuracy — the number a deployment sizing decision should use.
- `looseness(rib, k_wall) = ln(k_wall) / rib`: how much lower the certified bound sits than the wall.
- `preflight_report(..., r_ib=...)` now also returns `k_wall` and `looseness`, adds the `practical_wall`
  item and appends the wall to the verdict string.

### Changed
- `kv_storage_capacity` moved from `CANDIDATE` to `PARTIALLY VERIFIED`: the storage tier and the per-token
  flow mechanism were verified on associative recall, with the measured caveat that the formula is an
  optimistic floor (the constant is 1.5–4× high and the factor grows with the sequence length).
- `gauss_capacity` documented as verified end-to-end **with** a usability caveat (a rigorous upper bound
  that overestimates what a trained encoder uses by ~5× on the measured task).

## Earlier versions (0.1 – 0.4)

No changelog was kept. By 0.4.0 the package contained the ten modules that exist today: the VQ and
Gaussian bottlenecks, the capacity formulas and the min/sum wiring bounds, the two-tier
`conditional_capacity` reading, the quantization and distillation screens, the Fisher diagnostics, the
geometry probe, the training/evaluation glue with the Fano inversion, and the preflight checklist.
