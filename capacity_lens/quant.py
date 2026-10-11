"""Quantization bit-count lower bound: rate-distortion gives "how many bits to quantize to without losing task information".

Core formula: capacity upper bound under uniform quantization = d * log2(q) >= R_IB
=> minimum bits per dimension log2(q) >= R_IB / d
=> minimum number of quantization levels q_min = 2^(R_IB / d)
"""


class QuantCapacity:
    """Quantization-capacity matching: given the task information and dimension, gives the lower bound on the minimum number of quantization levels."""

    def min_bits_per_dim(self, R_ib, d):
        """Minimum bits per dimension log2(q).

        Args:
            R_ib: task-required information (bits, log2 base)
            d: bottleneck dimension
        Returns:
            R_ib / d
        """
        return R_ib / d

    def min_levels(self, R_ib, d):
        """Minimum number of quantization levels q_min = 2^(R_ib / d)."""
        return 2 ** (R_ib / d)

    def min_bits(self, R_ib, d):
        """Integer bit count: ceil(log2(q_min)), i.e. how many integer bits are needed per dimension at minimum."""
        import math
        return max(1, math.ceil(R_ib / d))

    def report(self, R_ib, d):
        """Print quantization recommendations."""
        bits_per_dim = self.min_bits_per_dim(R_ib, d)
        levels = self.min_levels(R_ib, d)
        int_bits = self.min_bits(R_ib, d)
        print(f"[quant] task information R_IB = {R_ib:.2f} bits, dim d = {d}")
        print(f"[quant] min {bits_per_dim:.3f} bits/dim -> quantize to at least {int_bits} bit")
        print(f"[quant] continuous level lower bound q_min = {levels:.1f} levels")
        return {"bits_per_dim": bits_per_dim, "int_bits": int_bits, "q_min": levels}
