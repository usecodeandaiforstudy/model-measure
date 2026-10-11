"""Distillation capacity lower bound: the student's capacity upper bound must be >= the task-required information R_IB.

Core formula: the student's discrete bottleneck capacity log K >= R_IB
=> minimum codebook size K_min = ceil(e^(R_IB))
"""

import math


class DistillCapacity:
    """Distillation-capacity matching: given the task information, gives the student's minimum capacity."""

    def min_codebook_size(self, R_ib):
        """Minimum codebook size K_min = ceil(e^(R_ib)).

        Args:
            R_ib: task-required information (natural logarithm base, nats)
        Returns:
            minimum codebook size (at least 2)
        """
        return max(2, math.ceil(math.exp(R_ib)))

    def min_capacity_nats(self, R_ib):
        """Student capacity lower bound = R_ib (natural units)."""
        return R_ib

    def report(self, R_ib):
        """Print distillation recommendations."""
        k_min = self.min_codebook_size(R_ib)
        print(f"[distill] task information R_IB = {R_ib:.2f} nats")
        print(f"[distill] student minimum codebook size K_min = {k_min}")
        return {"K_min": k_min}
