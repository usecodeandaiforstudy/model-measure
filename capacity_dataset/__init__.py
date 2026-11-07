"""capacity_dataset -- independent dataset decision reports built on capacity_lens (read-only)."""
from .codec import codec_report, latent_report
from .report import (__version__, build_report, dup_report, recommend, review_price,
                     saturation_report, selftest)

__all__ = ["__version__", "build_report", "dup_report", "recommend", "review_price",
           "saturation_report", "selftest", "codec_report", "latent_report"]
