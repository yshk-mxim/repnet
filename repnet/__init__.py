"""RepNet: Repeatable Networks — Deterministic Deep Learning.

Provides structured orthogonal weight initialization, deterministic batch ordering,
and verified bit-identical training for reproducible deep learning.
"""

__version__ = "1.0.0"

from repnet.init import deterministic_init
from repnet.models import ECGRepNetConformer, BaselineCNN
from repnet.batch import build_golden_ratio_batches, build_seeded_batches
from repnet.data import load_ptbxl

__all__ = [
    "deterministic_init",
    "ECGRepNetConformer",
    "BaselineCNN",
    "build_golden_ratio_batches",
    "build_seeded_batches",
    "load_ptbxl",
]
