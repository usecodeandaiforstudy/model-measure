"""Capacity Lens: an information-theoretic capacity diagnosis and deployment-matching toolkit for neural networks."""

from .bottleneck import VQBottleneck, GaussBottleneck
from .capacity import (
    discrete_capacity_logK,
    discrete_capacity_uniform,
    gauss_capacity,
    min_capacity_serial,
    sum_capacity_parallel,
    conditional_capacity,
    kv_storage_capacity,
)
from .monitor import TrainMonitor
from .quant import QuantCapacity
from .distill import DistillCapacity
from .fisher import FisherEffectiveDim, FisherArcLength, arc_length_step
from .training import (train_one_epoch, evaluate, entropy_of,
                       mutual_info_of, task_info_fano, joint_code)
from .geometry import (silhouette_by_label, separation_ratio,
                       vq_quantization_error, err_1nn, wall_geometry_probe)
from .preflight import (g0_gate, distill_plateau_ok, distill_gap_k_star,
                        excluded_set, coverage, false_exclusions,
                        measured_wall, k_wall_band, looseness, ratio_verdict, preflight_report)
from .compress import (balance_warning, task_info_floor, capacity_floors, veto, tightness_interval,
                       eps_for_rib, partition_retention, partition_ceiling, kmeans,
                       probe_lower_bound, selftest as compress_selftest, BIT_GRID,
                       quantise_weight, total_bits, average_bits, accuracy_at_bits,
                       plan_bit_allocation, MIN_RANK, rank_parameter_count,
                       uniform_rank_allocation, rank_plan_accuracy, plan_rank_allocation,
                       retention_curve, shortlist_rank_allocations, budget_ladder,
                       choose_operating_point, STRATEGY_DISCLAIMER, FLAT_TOL, REF_FLOOR)

__all__ = [
    "VQBottleneck",
    "GaussBottleneck",
    "discrete_capacity_logK",
    "discrete_capacity_uniform",
    "gauss_capacity",
    "min_capacity_serial",
    "sum_capacity_parallel",
    "conditional_capacity",
    "kv_storage_capacity",
    "TrainMonitor",
    "QuantCapacity",
    "DistillCapacity",
    "FisherEffectiveDim",
    "FisherArcLength",
    "arc_length_step",
    "train_one_epoch",
    "evaluate",
    "entropy_of",
    "mutual_info_of",
    "task_info_fano",
    "joint_code",
    "silhouette_by_label",
    "separation_ratio",
    "vq_quantization_error",
    "err_1nn",
    "wall_geometry_probe",
    "g0_gate",
    "distill_plateau_ok",
    "distill_gap_k_star",
    "excluded_set",
    "coverage",
    "false_exclusions",
    "measured_wall",
    "k_wall_band",
    "looseness",
    "ratio_verdict",
    "preflight_report",
    "balance_warning",
    "task_info_floor",
    "capacity_floors",
    "veto",
    "tightness_interval",
    "eps_for_rib",
    "partition_retention",
    "partition_ceiling",
    "kmeans",
    "probe_lower_bound",
    "compress_selftest",
    "BIT_GRID",
    "quantise_weight",
    "total_bits",
    "average_bits",
    "accuracy_at_bits",
    "plan_bit_allocation",
    "MIN_RANK",
    "rank_parameter_count",
    "uniform_rank_allocation",
    "rank_plan_accuracy",
    "plan_rank_allocation",
    "retention_curve",
    "shortlist_rank_allocations",
    "budget_ladder",
    "choose_operating_point",
    "STRATEGY_DISCLAIMER",
    "FLAT_TOL",
    "REF_FLOOR",
]

__version__ = "1.2.0"


# --- extension (1.1.0): loss-denominated constraint cost, memorisation capacity, subset representativeness,
# --- bit-side curves and encoder trade-offs. Separate modules; nothing above is modified.
from . import batch_delta, bit_curves, capacity_probe, cost, represent, tradeoff  # noqa: E402
from .batch_delta import (delta_summary, online_step_delta, per_batch_delta,  # noqa: E402
                          probe_signal, rank_batches, select_batches, track_probe)
from .cost import (alignment_prior, certify_lower, cg_solve_hinv_g, constraint_cost,  # noqa: E402
                   is_constrained_local_min)
from .capacity_probe import bisect_threshold, measure_capacity, p_fit  # noqa: E402
from .represent import align_report, calibrate_threshold, subset_rho_prime  # noqa: E402
from .bit_curves import activation_curve, pruning_curve  # noqa: E402
from .tradeoff import codec_tradeoff  # noqa: E402

__all__ = list(__all__) + [
    "batch_delta", "per_batch_delta", "delta_summary", "rank_batches",
    "select_batches", "online_step_delta", "track_probe",
    "bit_curves", "capacity_probe", "cost", "represent", "tradeoff",
    "cg_solve_hinv_g", "constraint_cost", "certify_lower", "is_constrained_local_min", "alignment_prior",
    "p_fit", "bisect_threshold", "measure_capacity",
    "calibrate_threshold", "subset_rho_prime", "align_report",
    "activation_curve", "pruning_curve", "codec_tradeoff",
]
