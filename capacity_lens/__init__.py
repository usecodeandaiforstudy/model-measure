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
                       plan_bit_allocation)

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
]

__version__ = "0.7.0"
